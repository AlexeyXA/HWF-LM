"""Stage 1: Measure rho(ell) — integrated fit; reproducible seed per W.

integrated_power_law accepts scalar or ndarray (required by scipy curve_fit).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import random
import subprocess

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from scipy.special import zeta
from tqdm import tqdm
import matplotlib.pyplot as plt

from hwf_lm.config import RhoResult, MeasureConfig
from hwf_lm.data import make_dataloader, load_tinystories_tokens


class WindowedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, window_size, max_len, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.emb.weight

    def _make_mask(self, N, device):
        idx = torch.arange(N, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        allowed = (diff >= 0) & (diff < self.window_size)
        return ~allowed

    def forward(self, x):
        B, N = x.shape
        pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.emb(x) + self.pos(pos)
        h = self.blocks(h, mask=self._make_mask(N, x.device))
        return self.lm_head(self.norm(h))


def integrated_power_law(W, rho_0, alpha, H_0):
    """L(W) = H_0 - rho_0 * sum_{ell=1..W} ell^{-alpha}.

    Accepts scalar or array W — scipy.optimize.curve_fit passes arrays.
    """
    W_arr = np.asarray(W, dtype=np.float64)
    scalar = W_arr.ndim == 0
    W_arr = np.atleast_1d(W_arr)
    out = np.empty_like(W_arr, dtype=np.float64)
    for i, w in enumerate(W_arr):
        W_int = int(round(float(w)))
        if W_int < 1:
            out[i] = float(H_0)
        else:
            ls = np.arange(1, W_int + 1, dtype=np.float64)
            out[i] = float(H_0) - float(rho_0) * float(np.sum(ls ** (-float(alpha))))
    return float(out[0]) if scalar else out


def unigram_entropy_nats(ids):
    counts = Counter(ids.tolist())
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def train_window_model(model, train_loader, val_loader, device, token_budget, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    tokens_seen = 0
    while tokens_seen < token_budget:
        for x, y in train_loader:
            if tokens_seen >= token_budget:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tokens_seen += x.numel()
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(crit(logits.view(-1, logits.size(-1)), y.view(-1)).item())
    return float(np.mean(losses))


def measure_rho(cfg: MeasureConfig) -> RhoResult:
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Stage 1 on {device}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    print(f"seed={cfg.seed}")
    if cfg.num_layers != 1:
        print(f"WARNING: num_layers={cfg.num_layers} => W_eff = 1+L*(W-1); prefer L=1")

    train_w, val_w, vocab_size = load_tinystories_tokens(
        num_stories=cfg.num_stories, seq_len=cfg.seq_len, seed=cfg.seed
    )
    all_ids = np.concatenate([train_w.reshape(-1), val_w.reshape(-1)])
    H_0 = unigram_entropy_nats(all_ids)
    print(f"H_0 (nats) = {H_0:.4f}")

    train_loader = make_dataloader(train_w, cfg.batch_size, shuffle=True)
    val_loader = make_dataloader(val_w, cfg.batch_size, shuffle=False, drop_last=False)

    windows = [w for w in cfg.windows if w <= cfg.max_window]
    losses = []
    for W in tqdm(windows, desc="windows"):
        eff = 1 + cfg.num_layers * max(W - 1, 0)
        print(f"\n[W={W:3d} W_eff={eff:3d}] budget={cfg.token_budget/1e6:.1f}M")
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        model = WindowedTransformer(
            vocab_size, cfg.d_model, cfg.num_layers, cfg.num_heads,
            window_size=W, max_len=cfg.seq_len,
        ).to(device)
        val_loss = train_window_model(
            model, train_loader, val_loader, device, cfg.token_budget, lr=cfg.lr
        )
        losses.append(val_loss)
        print(f"  W={W:3d} | Loss={val_loss:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        popt, _ = curve_fit(
            lambda W, r, a: integrated_power_law(W, r, a, H_0),
            np.array(windows, dtype=float),
            np.array(losses, dtype=float),
            p0=[1.0, 1.5],
            bounds=([1e-6, 1.05], [np.inf, 2.5]),
            method="trf",
            maxfev=20000,
        )
        rho_0, alpha = float(popt[0]), float(popt[1])
    except Exception as e:
        raise RuntimeError(f"Integrated fit failed: {e}") from e

    if rho_0 <= 0 or alpha <= 1.0:
        raise ValueError(f"Invalid fit: rho_0={rho_0}, alpha={alpha}")

    I_inf = float(rho_0 * zeta(alpha))
    pred = np.array([integrated_power_law(W, rho_0, alpha, H_0) for W in windows])
    residuals = np.array(losses) - pred
    max_rel = float(np.abs(residuals / (np.array(losses) + 1e-12)).max())

    result = RhoResult(
        rho_0=rho_0, alpha=alpha, H_0=float(H_0), I_inf=I_inf, max_rel_residual=max_rel
    )
    print("\n=== RESULTS (integrated fit) ===")
    print(f"  H_0={result.H_0:.4f} nats  rho_0={result.rho_0:.4f}  alpha={result.alpha:.4f}")
    print(f"  I_inf={result.I_inf:.4f} nats  max|res|={max_rel*100:.2f}%")
    print("Applicable:", result.is_applicable())

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(windows, losses, s=60, zorder=3, label="measured")
    Wg = np.linspace(max(1, min(windows)), max(windows), 200)
    axes[0].plot(Wg, [integrated_power_law(w, rho_0, alpha, H_0) for w in Wg], "r-",
                 label=f"fit alpha={alpha:.3f}")
    axes[0].axhline(H_0, color="gray", ls="--")
    axes[0].set_xlabel("W"); axes[0].set_ylabel("val CE (nats)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].bar(range(len(windows)), (residuals / (np.array(losses) + 1e-12)) * 100, color="C3")
    axes[1].axhline(0, color="k"); axes[1].axhline(5, color="red", ls="--", alpha=0.5)
    axes[1].axhline(-5, color="red", ls="--", alpha=0.5)
    axes[1].set_xticks(range(len(windows))); axes[1].set_xticklabels([str(w) for w in windows])
    plt.tight_layout(); plt.savefig(out_dir / "rho_fit.png", dpi=150); plt.close()

    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass

    detail = {
        "git_commit": git_commit,
        "seed": cfg.seed,
        "config": {
            "num_stories": cfg.num_stories,
            "seq_len": cfg.seq_len,
            "d_model": cfg.d_model,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "token_budget": cfg.token_budget,
            "max_window": cfg.max_window,
            "windows": list(windows),
            "batch_size": cfg.batch_size,
            "device": device,
        },
        "fit": {
            "rho_0": result.rho_0,
            "alpha": result.alpha,
            "H_0": result.H_0,
            "I_inf": result.I_inf,
            "max_rel_residual": result.max_rel_residual,
            "applicable": result.is_applicable(),
        },
        "per_window": {
            "W": list(windows),
            "val_ce_nats": [float(x) for x in losses],
            "pred_ce_nats": [float(x) for x in pred],
            "residual_abs": [float(x) for x in residuals],
            "residual_rel": [float(r / (l + 1e-12)) for r, l in zip(residuals, losses)],
        },
    }
    result.save(out_dir / "rho_result.json")
    with open(out_dir / "h1_detail.json", "w") as f:
        json.dump(detail, f, indent=2)
    print(f"Saved -> {out_dir}/rho_result.json, h1_detail.json, rho_fit.png")
    print(f"git_commit={git_commit} seed={cfg.seed}")
    return result

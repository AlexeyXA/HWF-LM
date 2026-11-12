"""Stage 1: Measure ρ(ℓ) via hard-window Transformers + power-law fit."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit
from scipy.special import zeta
from tqdm import tqdm
import matplotlib.pyplot as plt

from hwf_lm.config import RhoResult, MeasureConfig
from hwf_lm.data import get_dataloaders, make_dataloader, load_tinystories_tokens


class WindowedTransformer(nn.Module):
    """Small causal Transformer with hard attention window W."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 window_size: int, max_len: int, dropout: float = 0.1):
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
        self.max_len = max_len

    def _make_mask(self, N: int, device: torch.device) -> torch.Tensor:
        # SDPA boolean convention: True = ALLOWED to attend
        # For TransformerEncoder, mask True = blocked — we invert below
        idx = torch.arange(N, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        allowed = (diff >= 0) & (diff <= self.window_size)
        # TransformerEncoder expects True = blocked
        return ~allowed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.emb(x) + self.pos(pos)
        mask = self._make_mask(N, x.device)
        h = self.blocks(h, mask=mask)
        return self.lm_head(self.norm(h))


def unigram_entropy_nats(ids: np.ndarray) -> float:
    """H0 in nats (matches CrossEntropyLoss)."""
    counts = Counter(ids.tolist())
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def train_window_model(model, train_loader, val_loader, device, token_budget, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    tokens_seen = 0
    step = 0
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
            step += 1
    # val
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(crit(logits.view(-1, logits.size(-1)), y.view(-1)).item())
    return float(np.mean(losses))


def power_law(L, rho0, alpha):
    return rho0 * np.power(L, -alpha)


def fit_power_law(windows: List[int], losses: List[float], H0: float) -> Tuple[float, float, float, float]:
    """Fit I(L) = H0 - loss(L) ~ I_inf - C * L^{-alpha}; return rho0, alpha, I_inf, max_rel_residual."""
    L = np.array(windows, dtype=float)
    loss = np.array(losses, dtype=float)
    I = H0 - loss  # information gained
    # Fit residual entropy form: loss(L) ≈ I_inf_offset + rho0 * L^{-alpha}
    # Simpler: fit excess loss above asymptotic
    try:
        popt, _ = curve_fit(power_law, L, np.maximum(loss - loss.min() + 1e-6, 1e-6),
                            p0=[1.0, 1.2], bounds=([1e-6, 0.5], [100, 5]))
        rho0, alpha = float(popt[0]), float(popt[1])
        pred = power_law(L, rho0, alpha)
        residual = np.abs(pred - (loss - loss.min() + 1e-6)) / (np.abs(loss) + 1e-6)
        max_rel = float(np.max(residual))
    except Exception:
        rho0, alpha, max_rel = 1.0, 1.2, 1.0
    I_inf = float(H0 - loss.min())
    return rho0, alpha, I_inf, max_rel


def measure_rho(cfg: MeasureConfig) -> RhoResult:
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Stage 1 on {device}")

    train_w, val_w, vocab_size = load_tinystories_tokens(
        num_stories=cfg.num_stories, seq_len=cfg.seq_len, seed=42
    )
    all_ids = np.concatenate([train_w.reshape(-1), val_w.reshape(-1)])
    H0 = unigram_entropy_nats(all_ids)
    print(f"H0 (nats) = {H0:.4f}")

    train_loader = make_dataloader(train_w, cfg.batch_size, shuffle=True)
    val_loader = make_dataloader(val_w, cfg.batch_size, shuffle=False, drop_last=False)

    windows = list(cfg.windows)
    losses = []
    for W in tqdm(windows, desc="windows"):
        model = WindowedTransformer(
            vocab_size, cfg.d_model, cfg.num_layers, cfg.num_heads,
            window_size=W, max_len=cfg.seq_len,
        ).to(device)
        val_loss = train_window_model(
            model, train_loader, val_loader, device, cfg.token_budget, lr=cfg.lr
        )
        losses.append(val_loss)
        print(f"  W={W:4d}  val_loss={val_loss:.4f}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    rho0, alpha, I_inf, max_rel = fit_power_law(windows, losses, H0)
    result = RhoResult(rho_0=rho0, alpha=alpha, H_0=H0, I_inf=I_inf, max_rel_residual=max_rel)
    print(f"\nrho0={rho0:.4f} alpha={alpha:.4f} residual={max_rel*100:.2f}%")
    print(f"Applicable: {result.is_applicable()}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(windows, losses, "o-", label="val CE (nats)")
    Lplot = np.linspace(min(windows), max(windows), 100)
    ax.plot(Lplot, losses[-1] + power_law(Lplot, rho0, alpha) - power_law(windows[-1], rho0, alpha),
            "--", label=f"fit α={alpha:.2f}")
    ax.set_xlabel("window W")
    ax.set_ylabel("val loss (nats)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "rho_fit.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot → {plot_path}")

    result.save(out_dir / "rho_result.json")
    print(f"Saved result → {out_dir / 'rho_result.json'}")
    return result

#!/usr/bin/env python3
"""Matched-parameter baselines: HWF-LM vs Transformer vs PureSSM."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from hwf_lm.config import HWFConfig
from hwf_lm.models import HWF_LM
from hwf_lm.data import get_dataloaders
from hwf_lm.models.ssm import SSMRegion

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.emb.weight
    def forward(self, x, return_aux=False):
        B, N = x.shape
        pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.emb(x) + self.pos(pos)
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.blocks(h, mask=mask)
        logits = self.lm_head(self.norm(h))
        return (logits, torch.tensor(0.0, device=x.device)) if return_aux else logits
    def num_parameters(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.emb.weight.numel() if non_embedding else n

class PureSSM_LM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, ssm_state, max_len, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.region = SSMRegion(d_model, n_layers, ssm_state, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.emb.weight
    def forward(self, x, return_aux=False):
        B, N = x.shape
        pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.emb(x) + self.pos(pos)
        h = self.region(h)
        logits = self.lm_head(self.norm(h))
        return (logits, torch.tensor(0.0, device=x.device)) if return_aux else logits
    def num_parameters(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.emb.weight.numel() if non_embedding else n

def count_params(model, non_emb=True):
    if hasattr(model, "num_parameters"):
        return model.num_parameters(non_embedding=non_emb)
    return sum(p.numel() for p in model.parameters())

def find_transformer_layers(target, vocab_size, d_model, n_heads, max_len, tol=0.08):
    best_n, best_diff = 4, float("inf")
    for n in range(2, 16):
        m = TransformerLM(vocab_size, d_model, n_heads, n, max_len)
        diff = abs(count_params(m) - target) / max(target, 1)
        if diff < best_diff:
            best_diff, best_n = diff, n
        if diff < tol:
            return n
    return best_n

def find_ssm_layers(target, vocab_size, d_model, ssm_state, max_len, tol=0.08):
    best_n, best_diff = 4, float("inf")
    for n in range(2, 16):
        m = PureSSM_LM(vocab_size, d_model, n, ssm_state, max_len)
        diff = abs(count_params(m) - target) / max(target, 1)
        if diff < best_diff:
            best_diff, best_n = diff, n
        if diff < tol:
            return n
    return best_n

def get_lr(step, max_steps, lr, warmup):
    if step < warmup:
        return lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

@torch.no_grad()
def eval_loss(model, loader, device, crit):
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, return_aux=True)
        logits = out[0] if isinstance(out, tuple) else out
        losses.append(crit(logits.view(-1, logits.size(-1)), y.view(-1)).item())
    model.train()
    return float(np.mean(losses))

def train_one(model, name, train_loader, val_loader, device, max_steps, lr=3e-4, warmup=100, aux_coef=0.01, eval_every=200, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    step, best_val, history = 0, float("inf"), []
    t0 = time.time(); model.train()
    while step < max_steps:
        for x, y in train_loader:
            if step >= max_steps: break
            x, y = x.to(device), y.to(device)
            for g in opt.param_groups:
                g["lr"] = get_lr(step, max_steps, lr, warmup)
            out = model(x, return_aux=True)
            logits, aux = (out if isinstance(out, tuple) else (out, torch.tensor(0.0, device=device)))
            loss = crit(logits.view(-1, logits.size(-1)), y.view(-1)) + aux_coef * aux
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step > 0 and step % eval_every == 0:
                val = eval_loss(model, val_loader, device, crit)
                history.append({"step": step, "val_loss": val})
                if val < best_val: best_val = val
                print(f"  [{name}] step {step} | val={val:.4f}")
            step += 1
    return {"name": name, "params_M": count_params(model)/1e6, "best_val": best_val,
            "final_val": eval_loss(model, val_loader, device, crit),
            "wall_sec": time.time()-t0, "history": history, "seed": seed}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hwf-config", type=str, default=None)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--num-stories", type=int, default=5000)
    p.add_argument("--seeds", type=str, default="42")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="outputs/baselines")
    p.add_argument("--skip-ssm", action="store_true")
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Seeds: {seeds}")
    train_loader, val_loader, vocab_size = get_dataloaders(
        num_stories=args.num_stories, seq_len=args.seq_len, batch_size=args.batch_size, seed=42)
    if args.hwf_config and Path(args.hwf_config).exists():
        hwf_cfg = HWFConfig.load(args.hwf_config)
        hwf_cfg.d_model = args.d_model; hwf_cfg.n_heads = args.n_heads
        hwf_cfg.max_len = max(hwf_cfg.max_len, args.seq_len); hwf_cfg.vocab_size = vocab_size
    else:
        hwf_cfg = HWFConfig(ell_1=16, ell_2=64, ssm_state=32, nca_T=3, nca_k=3,
            n_layers_per_region=(2,1,1), d_model=args.d_model, n_heads=args.n_heads,
            vocab_size=vocab_size, max_len=args.seq_len, num_experts=4, moe_top_k=2)
    target = count_params(HWF_LM(hwf_cfg))
    print(f"HWF-LM params: {target/1e6:.2f}M")
    n_tf = find_transformer_layers(target, vocab_size, args.d_model, args.n_heads, args.seq_len)
    models_to_run = {
        "HWF-LM": lambda: HWF_LM(hwf_cfg),
        "Transformer": lambda: TransformerLM(vocab_size, args.d_model, args.n_heads, n_tf, args.seq_len),
    }
    if not args.skip_ssm:
        n_ssm = find_ssm_layers(target, vocab_size, args.d_model, hwf_cfg.ssm_state, args.seq_len)
        models_to_run["PureSSM"] = lambda: PureSSM_LM(vocab_size, args.d_model, n_ssm, hwf_cfg.ssm_state, args.seq_len)
    all_results = []
    for seed in seeds:
        print(f"\n===== Seed {seed} =====")
        for name, factory in models_to_run.items():
            print(f"\nTraining {name}...")
            result = train_one(factory(), name, train_loader, val_loader, device, args.max_steps, seed=seed)
            all_results.append(result)
            print(f"  -> best_val={result['best_val']:.4f} wall={result['wall_sec']:.0f}s")
    by_name = {}
    for r in all_results:
        by_name.setdefault(r["name"], []).append(r)
    summary = {}
    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    for name, runs in by_name.items():
        vals = [r["best_val"] for r in runs]
        walls = [r["wall_sec"] for r in runs]
        summary[name] = {"mean_best_val": float(np.mean(vals)), "std_best_val": float(np.std(vals)),
            "mean_wall_sec": float(np.mean(walls)), "params_M": runs[0]["params_M"], "n_seeds": len(runs)}
        print(f"{name:<12} val={np.mean(vals):.4f}+/-{np.std(vals):.4f} wall={np.mean(walls):.0f}s params={runs[0]['params_M']:.2f}M")
    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump({"runs": all_results, "summary": summary}, f, indent=2)
    print(f"Saved -> {out_dir / 'baseline_results.json'}")

if __name__ == "__main__":
    main()

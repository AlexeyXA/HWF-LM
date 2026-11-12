"""Training utilities for HWF-LM."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from hwf_lm.config import HWFConfig, TrainConfig
from hwf_lm.data import get_dataloaders


def get_lr(step: int, max_steps: int, lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, crit) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, return_aux=True)
        logits = out[0] if isinstance(out, tuple) else out
        losses.append(crit(logits.view(-1, logits.size(-1)), y.view(-1)).item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train(
    model: nn.Module,
    train_cfg: TrainConfig,
    hwf_cfg: Optional[HWFConfig] = None,
) -> dict:
    device = train_cfg.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    crit = nn.CrossEntropyLoss()

    train_loader, val_loader, vocab_size = get_dataloaders(
        num_stories=train_cfg.num_stories,
        seq_len=train_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        seed=train_cfg.seed,
    )
    if hwf_cfg is not None and hwf_cfg.vocab_size != vocab_size:
        print(f"Note: vocab_size from data={vocab_size}, config={hwf_cfg.vocab_size}")

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    best_val = float("inf")
    t0 = time.time()
    model.train()

    while step < train_cfg.max_steps:
        for x, y in train_loader:
            if step >= train_cfg.max_steps:
                break
            x, y = x.to(device), y.to(device)
            for g in opt.param_groups:
                g["lr"] = get_lr(step, train_cfg.max_steps, train_cfg.lr, train_cfg.warmup_steps)

            out = model(x, return_aux=True)
            if isinstance(out, tuple):
                logits, aux = out
            else:
                logits, aux = out, torch.tensor(0.0, device=device)
            loss = crit(logits.view(-1, logits.size(-1)), y.view(-1)) + train_cfg.aux_loss_coef * aux

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()

            if step > 0 and step % train_cfg.eval_every == 0:
                val = evaluate(model, val_loader, device, crit)
                print(f"step {step} | train_loss={loss.item():.4f} | val={val:.4f} | lr={opt.param_groups[0]['lr']:.2e}")
                if val < best_val:
                    best_val = val
                    torch.save(
                        {"model": model.state_dict(), "step": step, "val": val},
                        out_dir / "best.pt",
                    )

            if step > 0 and step % train_cfg.save_every == 0:
                torch.save(
                    {"model": model.state_dict(), "step": step},
                    out_dir / f"ckpt_{step}.pt",
                )
            step += 1

    elapsed = time.time() - t0
    final_val = evaluate(model, val_loader, device, crit)
    print(f"Done. best_val={best_val:.4f} final_val={final_val:.4f} wall={elapsed:.0f}s")
    return {"best_val": best_val, "final_val": final_val, "wall_sec": elapsed}

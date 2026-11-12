"""Measure real relative costs of HWF-LM mechanisms."""

from __future__ import annotations

import time
import gc
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import json
from pathlib import Path

import torch
import torch.nn as nn

from hwf_lm.models.ssm import SSMRegion
from hwf_lm.models.nca import NCARegion
from hwf_lm.models.attention import AttnRegion


@dataclass
class CostResult:
    name: str
    latency_ms: float
    latency_std_ms: float
    peak_mem_mb: float
    tokens_per_sec: float
    relative_cost: float
    seq_len: int
    batch_size: int
    d_model: int
    n_layers: int
    device: str


def _synchronize(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_memory_mb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def _reset_peak(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


@torch.no_grad()
def benchmark_module(module, name, batch_size, seq_len, d_model, device, n_warmup=10, n_reps=30, n_layers=1):
    module = module.to(device).eval()
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    for _ in range(n_warmup):
        _ = module(x)
    _synchronize(device)
    _reset_peak(device)
    times = []
    for _ in range(n_reps):
        _synchronize(device)
        t0 = time.perf_counter()
        _ = module(x)
        _synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    peak = _peak_memory_mb(device)
    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    tokens = batch_size * seq_len
    tps = tokens / (mean_ms / 1000.0) if mean_ms > 0 else 0.0
    return CostResult(name, mean_ms, std_ms, peak, tps, 1.0, seq_len, batch_size, d_model, n_layers, device)


def measure_all_costs(
    d_model=256, n_heads=8, seq_len=256, batch_size=8, ssm_state=64,
    nca_window=3, nca_T=3, local_window=128, n_global=4, device=None,
    n_warmup=10, n_reps=30,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Measuring costs on {device} | d_model={d_model}, seq_len={seq_len}, bs={batch_size}")

    modules = {
        "SSM": SSMRegion(d_model, n_layers=1, state_size=ssm_state),
        "NCA": NCARegion(d_model, n_layers=1, window=nca_window, T=nca_T),
        "SparseAttn": AttnRegion(d_model, n_layers=1, num_heads=n_heads, local_window=local_window, n_global=n_global),
    }

    class DenseAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
        def forward(self, x):
            h = self.norm(x)
            out, _ = self.attn(h, h, h, need_weights=False)
            return x + out

    modules["DenseAttn"] = DenseAttn()
    results = {}
    for name, mod in modules.items():
        print(f"  benchmarking {name}...")
        results[name] = benchmark_module(mod, name, batch_size, seq_len, d_model, device, n_warmup, n_reps)

    min_lat = min(r.latency_ms for r in results.values())
    for r in results.values():
        r.relative_cost = r.latency_ms / min_lat if min_lat > 0 else 1.0
    return results


def print_cost_table(results):
    print("\n" + "=" * 78)
    print(f"{'Mechanism':<12} {'lat (ms)':>10} {'±std':>8} {'mem (MB)':>10} {'tok/s':>10} {'rel cost':>10}")
    print("-" * 78)
    for name, r in results.items():
        print(f"{name:<12} {r.latency_ms:10.2f} {r.latency_std_ms:8.2f} {r.peak_mem_mb:10.1f} {r.tokens_per_sec:10.0f} {r.relative_cost:10.2f}")
    print("=" * 78)
    ref = results.get("SparseAttn") or results.get("DenseAttn")
    if ref and ref.relative_cost > 0:
        c_S = results["SSM"].relative_cost / ref.relative_cost
        c_N = results["NCA"].relative_cost / ref.relative_cost
        c_A = 1.0
        print(f"\nSuggested Stage-2 coefficients (relative to SparseAttn):")
        print(f"  c_S (SSM)  = {c_S:.4f}")
        print(f"  c_N (NCA)  = {c_N:.4f}")
        print(f"  c_A (Attn) = {c_A:.4f}")
        return {"c_S": c_S, "c_N": c_N, "c_A": c_A}
    return {}


def save_costs(results, coeffs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"results": {k: asdict(v) for k, v in results.items()}, "coefficients": coeffs}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved → {path}")

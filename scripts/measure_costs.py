#!/usr/bin/env python3
"""Measure relative costs of SSM / NCA / SparseAttn / DenseAttn."""
from __future__ import annotations
import argparse
from pathlib import Path
from hwf_lm.measure.costs import measure_all_costs, print_cost_table, save_costs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--ssm-state", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--output", type=str, default="outputs/costs.json")
    args = p.parse_args()
    results = measure_all_costs(
        d_model=args.d_model, n_heads=args.n_heads, seq_len=args.seq_len,
        batch_size=args.batch_size, ssm_state=args.ssm_state, device=args.device,
        n_warmup=args.warmup, n_reps=args.reps,
    )
    coeffs = print_cost_table(results)
    save_costs(results, coeffs, args.output)
    print("\nNext: feed these coefficients into Stage 2:")
    print(f"  python scripts/stage2_config.py --c-S {coeffs.get('c_S', 0.05):.4f} "
          f"--c-N {coeffs.get('c_N', 0.2):.4f} --c-A {coeffs.get('c_A', 1.0):.4f}")

if __name__ == "__main__":
    main()

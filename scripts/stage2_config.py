#!/usr/bin/env python3
"""Stage 2: Build HWFConfig from rho_result.json via water-filling."""
from __future__ import annotations
import argparse
from pathlib import Path
from hwf_lm.config import RhoResult
from hwf_lm.measure.thresholds import calculate_thresholds

def main():
    parser = argparse.ArgumentParser(description="HWF-LM Stage 2: auto-config")
    parser.add_argument("--rho", type=str, default="outputs/stage1/rho_result.json")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--output", type=str, default="outputs/hwf_config.json")
    parser.add_argument("--c-S", type=float, default=0.05)
    parser.add_argument("--c-N", type=float, default=0.2)
    parser.add_argument("--c-A", type=float, default=1.0)
    args = parser.parse_args()
    rho = RhoResult.load(args.rho)
    print(f"Loaded rho: alpha={rho.alpha:.4f}, residual={rho.max_rel_residual*100:.2f}%")
    if not rho.is_applicable():
        print("STOP: power law fails or alpha out of range. HWF-LM not applicable.")
        return
    config = calculate_thresholds(
        rho, max_len=args.max_len, d_model=args.d_model, n_heads=args.n_heads,
        c_S=args.c_S, c_N=args.c_N, c_A=args.c_A,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    config.save(out)
    print(f"\nConfig saved -> {out}")
    print(config)

if __name__ == "__main__":
    main()

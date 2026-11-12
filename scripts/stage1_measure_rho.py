#!/usr/bin/env python3
"""Stage 1: Measure rho(ell) on TinyStories."""

from __future__ import annotations

import argparse
from hwf_lm.config import MeasureConfig
from hwf_lm.measure.rho import measure_rho


def main():
    parser = argparse.ArgumentParser(description="HWF-LM Stage 1: measure rho(ell)")
    parser.add_argument("--num-stories", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Use 1 so nominal W equals effective context")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=2_000_000)
    parser.add_argument("--max-window", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="outputs/stage1")
    args = parser.parse_args()

    cfg = MeasureConfig(
        num_stories=args.num_stories,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        token_budget=args.token_budget,
        max_window=args.max_window,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(f"Stage 1 config: seq_len={cfg.seq_len} num_layers={cfg.num_layers} "
          f"max_window={cfg.max_window} budget={cfg.token_budget}")
    result = measure_rho(cfg)
    print("\n=== SAVE THIS ===")
    print(result)


if __name__ == "__main__":
    main()

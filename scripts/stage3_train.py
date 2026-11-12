#!/usr/bin/env python3
"""Stage 3: Train HWF-LM."""
from __future__ import annotations
import argparse
from hwf_lm.config import HWFConfig, TrainConfig
from hwf_lm.models.hwf_lm import HWF_LM
from hwf_lm.train import train

def main():
    parser = argparse.ArgumentParser(description="HWF-LM Stage 3: train")
    parser.add_argument("--config", type=str, default="outputs/hwf_config.json")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-stories", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="outputs/stage3")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    hwf_cfg = HWFConfig.load(args.config)
    hwf_cfg.max_len = max(hwf_cfg.max_len, args.seq_len)
    train_cfg = TrainConfig(
        batch_size=args.batch_size, seq_len=args.seq_len, lr=args.lr,
        max_steps=args.max_steps, num_stories=args.num_stories,
        device=args.device, output_dir=args.output_dir,
    )
    print("HWF Config:", hwf_cfg)
    print("Train Config:", train_cfg)
    model = HWF_LM(hwf_cfg)
    train(model, train_cfg, hwf_cfg)

if __name__ == "__main__":
    main()

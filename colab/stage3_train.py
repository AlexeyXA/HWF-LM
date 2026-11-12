# ============================================================
# STAGE 3 — train HWF-LM (Colab cell)
# Prefer: python scripts/stage3_train.py --config outputs/hwf_config.json
# ============================================================
print("For modular package run:")
print("  !python scripts/stage3_train.py \\\n    --config outputs/hwf_config.json \\\n    --max-steps 3000 --seq-len 256 --batch-size 16")
print()
print("Or matched baselines:")
print("  !python scripts/run_matched_baselines.py \\\n    --hwf-config outputs/hwf_config.json \\\n    --max-steps 2000 --seeds 42,43,44")

# ============================================================
# STAGE 1 — measure rho(ell) on TinyStories (Colab cell)
# Uses corrected causal mask (True=allowed) and nats units.
# Prefer: python scripts/stage1_measure_rho.py
# ============================================================
print("For modular package run:")
print("  !pip install -q tiktoken datasets matplotlib scipy tqdm pyyaml einops")
print("  !git clone https://github.com/AlexeyXA/HWF-LM.git")
print("  %cd HWF-LM")
print("  !python scripts/stage1_measure_rho.py --output-dir outputs/stage1")
print()
print("Or import from the installed package after cloning.")
print("Gate: residual <= 5% and alpha in (1.05, 2.5)")

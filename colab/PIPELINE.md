# HWF-LM — correct experiment order

## 0. Install (Colab T4)

```python
!pip install -q tiktoken datasets matplotlib scipy tqdm pyyaml einops
```

## 1. Measure real mechanism costs

```bash
python scripts/measure_costs.py --seq-len 256 --batch-size 8 --d-model 256 --output outputs/costs.json
```

## 2. Stage 1 — spectrum ρ(ℓ)

```bash
python scripts/stage1_measure_rho.py --output-dir outputs/stage1
```

**Gate:** residuals ≤ 5% and α ∈ (1.05, 2.5). Else STOP.

## 3. Stage 2 with measured c_*

```bash
python scripts/stage2_config.py --rho outputs/stage1/rho_result.json --c-S <from costs> --c-N <from costs> --c-A 1.0 --output outputs/hwf_config.json
```

## 4. Matched baselines

```bash
python scripts/run_matched_baselines.py --hwf-config outputs/hwf_config.json --max-steps 2000 --seeds 42,43,44
```

## 5. Interpretation only from step 4 numbers

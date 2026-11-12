# HWF-LM

**Hierarchical Water-Filling Language Model**

Decoder-only LM whose region order is driven by the measured task spectrum ρ(ℓ).

## Scientific pipeline (required order)

1. `scripts/measure_costs.py` → real c_S, c_N, c_A on target hardware
2. Stage 1 — measure ρ(ℓ) with **integrated** power-law fit
3. Stage 2 — thresholds using **all three** costs c_S, c_N, c_A
4. `scripts/run_matched_baselines.py` — matched params + seeds **before** init
5. Only then: any claims about speedup

See `colab/PIPELINE.md`.

## Stage 1 math (important)

Hypothesis:

```
ρ(ℓ) = ρ₀ · ℓ^{-α}
L(W) = H₀ - ρ₀ · Σ_{ℓ=1..W} ℓ^{-α}
```

Fit uses **integrated** form (`integrated_power_law`), not residual ~ W^{-β} with β treated as α.

Asymptotically L(W)-L_∞ ~ W^{1-α}, so β = α-1 if one fits residual power only — do not confuse them.

I_∞ = ρ₀ · ζ(α) (extrapolated), not H₀ - min_W L(W).

## v0.2 P0 fixes

- Story-level train/val split (no window leakage across stories)
- Water-filling uses c_N; missing roots raise instead of silent defaults
- Matched baselines: `torch.manual_seed(seed)` **before** `factory()`
- Causal masks / NCA / nats units (from v0.1)

## Still open (not claimed solved)

- Sparse Attn still dense N×N mask + SDPA (no FLOP guarantee)
- ell_1/ell_2 not yet hard-constraining attention ranges inside forward
- Full experimental validation of hypotheses 1–3 pending Colab runs

## Status

| Area | Status |
|------|--------|
| Causal correctness | OK |
| Stage 1 math (integrated fit) | OK in local package; verify on clone |
| Data split | OK |
| Water-filling c_N | OK |
| Seed reproducibility | OK |
| True sparse compute | Not yet |
| Matched empirical win | Not yet |

## License

MIT

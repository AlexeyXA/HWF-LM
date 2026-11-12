# HWF-LM

**Hierarchical Water-Filling Language Model**

Decoder-only LM whose region order is driven by the measured task spectrum ρ(ℓ).

## Scientific pipeline (required order)

1. `scripts/measure_costs.py` → real c_S, c_N, c_A on target hardware
2. Stage 1 — measure ρ(ℓ) (corrected causal mask)
3. Stage 2 — water-filling with measured costs
4. `scripts/run_matched_baselines.py` — HWF-LM vs Transformer vs PureSSM
5. Only then: any claims about speedup / Ω(n²)

See `colab/PIPELINE.md` for Colab T4 instructions.

## Status

- Causal masks verified (SDPA True=allowed)
- NCA is causal
- Causality unit tests included
- Cost measurement + matched baselines ready

## License

MIT

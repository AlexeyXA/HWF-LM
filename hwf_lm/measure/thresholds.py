"""Stage 2: ell_1, ell_2 from measured rho. Uses c_S, c_N, c_A. Missing roots raise."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from hwf_lm.config import RhoResult, HWFConfig


def calculate_thresholds(
    rho: RhoResult,
    ssm_state: int = 64,
    nca_k: int = 3,
    c_S: float = 0.05,
    c_N: float = 0.2,
    c_A: float = 1.0,
    max_len: int = 2048,
    d_model: int = 256,
    n_heads: int = 8,
    strict: bool = True,
) -> HWFConfig:
    rho_0, alpha = rho.rho_0, rho.alpha

    def rho_fn(L: float) -> float:
        return rho_0 * (max(L, 1.0) ** (-alpha))

    def eq_ssm(L: float) -> float:
        return c_S * np.exp(-L / max(ssm_state, 1)) - rho_fn(L)

    def eq_nca(L: float) -> float:
        return c_N / (1.0 + L / max(nca_k, 1)) - rho_fn(L)

    def eq_attn(L: float) -> float:
        return rho_fn(L) - c_A

    def solve(fn, lo, hi, name: str) -> float:
        try:
            if fn(lo) * fn(hi) > 0:
                raise ValueError(f"no sign change on [{lo},{hi}]")
            return float(brentq(fn, lo, hi))
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"No root for {name}: {e}. Check c_S={c_S}, c_N={c_N}, c_A={c_A}, alpha={alpha}."
                ) from e
            print(f"WARNING: no root for {name}, heuristic")
            return lo if "ell_1" in name else (lo + hi) / 2

    l1 = solve(eq_ssm, 1.0, min(1000.0, float(max_len)), "ell_1 (SSM)")
    try:
        l2 = solve(eq_nca, max(l1 + 1e-3, 1.0), float(max_len), "ell_2 (NCA)")
    except RuntimeError:
        l2 = solve(eq_attn, max(l1 + 1e-3, 1.0), float(max_len), "ell_2 (Attn fallback)")

    if l2 <= l1:
        if strict:
            raise RuntimeError(f"Invalid: ell_2={l2:.2f} <= ell_1={l1:.2f}")
        l2 = min(float(max_len), l1 * 2 + 1)

    if l1 > 4 * ssm_state:
        ssm_state = int(np.ceil(l1 / 2))

    n_ssm = min(4, max(1, int(np.ceil(l1 / max(ssm_state, 1)))))
    gap = max(0.0, l2 - l1)
    nca_T = min(50, max(3, int(np.ceil(gap / max(nca_k, 1)))))
    n_nca = min(2, max(1, int(np.ceil(gap / (nca_T * nca_k + 1e-6)))))
    n_attn = 1 if (max_len - l2) < 100 else 2

    print(f"ell_1={l1:.2f}, ell_2={l2:.2f} | c_S={c_S:.4f} c_N={c_N:.4f} c_A={c_A:.4f}")
    print(f"Layers: SSM={n_ssm}, NCA={n_nca} (T={nca_T}), Attn={n_attn}")

    return HWFConfig(
        ell_1=int(np.ceil(l1)), ell_2=int(np.ceil(l2)),
        ssm_state=ssm_state, nca_T=nca_T, nca_k=nca_k,
        n_layers_per_region=(n_ssm, n_nca, n_attn),
        d_model=d_model, n_heads=n_heads, max_len=max_len,
    )

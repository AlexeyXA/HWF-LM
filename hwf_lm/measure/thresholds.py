"""Stage 2 thresholds with multi-root scan; pick smallest positive SSM root."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import List, Optional
from hwf_lm.config import RhoResult, HWFConfig


def _find_roots(fn, lo: float, hi: float, n_grid: int = 64) -> List[float]:
    xs = np.linspace(lo, hi, n_grid)
    ys = np.array([fn(float(x)) for x in xs])
    roots = []
    for i in range(len(xs) - 1):
        if ys[i] == 0:
            roots.append(float(xs[i]))
        elif ys[i] * ys[i + 1] < 0:
            try:
                roots.append(float(brentq(fn, float(xs[i]), float(xs[i + 1]))))
            except ValueError:
                pass
    cleaned = []
    for r in sorted(roots):
        if not cleaned or abs(r - cleaned[-1]) > 1e-3:
            cleaned.append(r)
    return cleaned


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

    ssm_roots = _find_roots(eq_ssm, 1.0, min(2000.0, float(max_len)), n_grid=128)
    if not ssm_roots:
        if strict:
            raise RuntimeError(
                f"No SSM root. c_S={c_S}, alpha={alpha}, rho_0={rho_0}. "
                f"f(1)={eq_ssm(1):.4g}"
            )
        l1 = 1.0
        print("WARNING: no SSM root, l1=1")
    else:
        l1 = ssm_roots[0]
        if len(ssm_roots) > 1:
            print(f"Note: multiple SSM roots {ssm_roots}; using smallest l1={l1:.3f}")

    def first_above(fn) -> Optional[float]:
        roots = _find_roots(fn, max(l1 + 1e-3, 1.0), float(max_len), n_grid=128)
        return roots[0] if roots else None

    l2 = first_above(eq_nca)
    if l2 is None:
        l2 = first_above(eq_attn)
    if l2 is None:
        if strict:
            raise RuntimeError(f"No ell_2 root above l1={l1:.3f}. c_N={c_N}, c_A={c_A}")
        l2 = min(float(max_len), l1 * 2 + 1)
        print(f"WARNING: no ell_2 root, heuristic l2={l2}")

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

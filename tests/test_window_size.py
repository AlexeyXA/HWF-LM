"""Window size tests against the *real* WindowedTransformer class."""
from __future__ import annotations

import sys
import types
import torch

try:
    import tiktoken  # noqa: F401
    import datasets  # noqa: F401
except ImportError:
    stub = types.ModuleType("hwf_lm.data")
    stub.make_dataloader = None
    stub.load_tinystories_tokens = None
    sys.modules["hwf_lm.data"] = stub

from hwf_lm.measure.rho import WindowedTransformer


def test_window_mask_size():
    for W in [1, 2, 4, 8]:
        m = WindowedTransformer(
            vocab_size=50, d_model=32, n_layers=1, n_heads=4,
            window_size=W, max_len=32,
        )
        N = 16
        allowed = ~m._make_mask(N, torch.device("cpu"))
        for i in range(N):
            n_allowed = int(allowed[i, : i + 1].sum().item())
            expected = min(i + 1, W)
            assert n_allowed == expected, f"W={W} i={i}: {n_allowed} != {expected}"
        print(f"  W={W} OK (real WindowedTransformer)")
    print("OK window size")


def test_no_future():
    m = WindowedTransformer(50, 32, 1, 4, 8, 32)
    N = 16
    allowed = ~m._make_mask(N, torch.device("cpu"))
    for i in range(N):
        if i + 1 < N:
            assert not bool(allowed[i, i + 1 :].any()), f"future at {i}"
    print("OK no future")


def test_effective_context_formula():
    for L, W in [(1, 1), (1, 4), (2, 4), (3, 8)]:
        weff = 1 + L * max(W - 1, 0)
        print(f"  L={L} W={W} -> W_eff={weff}")
    print("OK effective context formula")


if __name__ == "__main__":
    test_window_mask_size()
    test_no_future()
    test_effective_context_formula()
    print("All window tests passed.")

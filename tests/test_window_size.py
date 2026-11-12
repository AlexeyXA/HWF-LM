"""Window size must admit exactly W causal positions."""
import torch
from hwf_lm.measure.rho import WindowedTransformer

def test_window_mask_size():
    for W in [1, 2, 4, 8]:
        m = WindowedTransformer(50, 32, 1, 4, W, 32)
        N = 16
        blocked = m._make_mask(N, torch.device("cpu"))
        allowed = ~blocked
        if allowed.dim() > 2:
            allowed = allowed.squeeze()
        for i in range(N):
            n_allowed = int(allowed[i, : i + 1].sum().item())
            expected = min(i + 1, W)
            assert n_allowed == expected, f"W={W} i={i}: {n_allowed} != {expected}"
        print(f"  W={W} OK")
    print("OK window size")

def test_no_future():
    m = WindowedTransformer(50, 32, 1, 4, 8, 32)
    N = 16
    allowed = ~m._make_mask(N, torch.device("cpu"))
    if allowed.dim() > 2:
        allowed = allowed.squeeze()
    for i in range(N):
        if i + 1 < N:
            assert not bool(allowed[i, i + 1 :].any())
    print("OK no future")

if __name__ == "__main__":
    test_window_mask_size(); test_no_future()
    print("All window tests passed.")

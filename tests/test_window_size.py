"""Window size W admits exactly W causal positions (self + W-1 past)."""
import torch
import torch.nn as nn

class _MiniWindow(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
    def _make_mask(self, N, device):
        idx = torch.arange(N, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        allowed = (diff >= 0) & (diff < self.window_size)
        return ~allowed  # TransformerEncoder: True=blocked

def test_window_mask_size():
    for W in [1, 2, 4, 8]:
        m = _MiniWindow(W)
        N = 16
        allowed = ~m._make_mask(N, torch.device("cpu"))
        for i in range(N):
            n_allowed = int(allowed[i, : i + 1].sum().item())
            expected = min(i + 1, W)
            assert n_allowed == expected, f"W={W} i={i}: {n_allowed} != {expected}"
        print(f"  W={W} OK")
    print("OK window size")

def test_no_future():
    m = _MiniWindow(8)
    N = 16
    allowed = ~m._make_mask(N, torch.device("cpu"))
    for i in range(N):
        if i + 1 < N:
            assert not bool(allowed[i, i + 1 :].any())
    print("OK no future")

if __name__ == "__main__":
    test_window_mask_size(); test_no_future()
    print("All window tests passed.")

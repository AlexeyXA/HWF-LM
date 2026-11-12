"""Causality tests: future tokens must not affect past logits."""

import torch
from hwf_lm.config import HWFConfig
from hwf_lm.models import HWF_LM
from hwf_lm.models.attention import SparseAttention
from hwf_lm.models.nca import NCAOnlyLayer


def test_sparse_attention_causal():
    attn = SparseAttention(d_model=32, num_heads=4, local_window=8, n_global=2)
    x = torch.randn(2, 16, 32)
    y1 = attn(x)
    x2 = x.clone()
    x2[:, 10:, :] = torch.randn_like(x2[:, 10:, :])
    y2 = attn(x2)
    assert torch.allclose(y1[:, :10], y2[:, :10], atol=1e-5), "Future leak in SparseAttention"
    print("OK SparseAttention causal")


def test_nca_causal():
    nca = NCAOnlyLayer(d_model=32, window=3, T=2)
    x = torch.randn(2, 16, 32)
    y1 = nca(x)
    x2 = x.clone()
    x2[:, 10:, :] = torch.randn_like(x2[:, 10:, :])
    y2 = nca(x2)
    assert torch.allclose(y1[:, :10], y2[:, :10], atol=1e-5), "Future leak in NCA"
    print("OK NCA causal")


def test_hwf_lm_causal():
    cfg = HWFConfig(
        ell_1=8, ell_2=16, ssm_state=16, nca_T=2, nca_k=2,
        n_layers_per_region=(1, 1, 1),
        d_model=32, n_heads=4, vocab_size=100, max_len=32,
        num_experts=4, moe_top_k=2, local_window=8, n_global_tokens=2,
    )
    model = HWF_LM(cfg)
    model.eval()
    x = torch.randint(0, 100, (2, 20))
    with torch.no_grad():
        logits1, _ = model(x)
        x2 = x.clone()
        x2[:, 12:] = torch.randint(0, 100, (2, 8))
        logits2, _ = model(x2)
    assert torch.allclose(logits1[:, :12], logits2[:, :12], atol=1e-4), "Future leak in HWF_LM"
    print("OK HWF_LM causal")


if __name__ == "__main__":
    test_sparse_attention_causal()
    test_nca_causal()
    test_hwf_lm_causal()
    print("All causality tests passed.")

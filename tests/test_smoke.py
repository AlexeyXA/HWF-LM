"""Smoke tests — run without GPU / long training."""
import torch
from hwf_lm.config import HWFConfig, RhoResult
from hwf_lm.models import HWF_LM
from hwf_lm.measure.thresholds import calculate_thresholds

def test_model_forward():
    cfg = HWFConfig(
        ell_1=16, ell_2=64, ssm_state=32, nca_T=3, nca_k=3,
        n_layers_per_region=(1, 1, 1),
        d_model=64, n_heads=4, vocab_size=1000, max_len=128,
        num_experts=4, moe_top_k=2, local_window=32, n_global_tokens=2,
    )
    model = HWF_LM(cfg)
    x = torch.randint(0, 1000, (2, 64))
    logits, aux = model(x)
    assert logits.shape == (2, 64, 1000)
    assert aux.ndim == 0
    print("OK forward pass, params =", model.num_parameters() / 1e6, "M")

def test_thresholds():
    rho = RhoResult(rho_0=1.2, alpha=1.4, H_0=4.5, I_inf=3.0, max_rel_residual=0.03)
    assert rho.is_applicable()
    cfg = calculate_thresholds(rho, max_len=512, d_model=128, n_heads=4)
    assert cfg.ell_1 >= 1
    assert cfg.ell_2 > cfg.ell_1
    print("OK thresholds:", cfg.n_layers_per_region)

if __name__ == "__main__":
    test_model_forward()
    test_thresholds()
    print("All smoke tests passed.")

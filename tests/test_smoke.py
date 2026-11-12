"""Smoke tests."""
import torch
from hwf_lm.config import HWFConfig, RhoResult
from hwf_lm.models import HWF_LM
from hwf_lm.measure.thresholds import calculate_thresholds

def test_model_forward():
    cfg = HWFConfig(
        ell_1=16, ell_2=64, ssm_state=32, nca_T=3, nca_k=3,
        n_layers_per_region=(1, 1, 1), d_model=64, n_heads=4,
        vocab_size=1000, max_len=128, num_experts=4, moe_top_k=2,
        local_window=32, n_global_tokens=2,
    )
    model = HWF_LM(cfg)
    x = torch.randint(0, 1000, (2, 64))
    logits, aux = model(x)
    assert logits.shape == (2, 64, 1000)
    print("OK forward", model.num_parameters() / 1e6, "M")

def test_thresholds_success():
    rho = RhoResult(rho_0=0.5, alpha=1.3, H_0=4.5, I_inf=2.0, max_rel_residual=0.03)
    cfg = calculate_thresholds(rho, max_len=512, c_S=2.0, c_N=0.5, c_A=0.01, strict=True)
    assert cfg.ell_2 > cfg.ell_1 >= 1
    print("OK thresholds", cfg.ell_1, cfg.ell_2)

def test_thresholds_failure():
    rho = RhoResult(rho_0=1.2, alpha=1.4, H_0=4.5, I_inf=3.0, max_rel_residual=0.03)
    try:
        calculate_thresholds(rho, max_len=512, c_S=1e-6, c_N=1e-6, c_A=1e6, strict=True)
        ok = False
    except RuntimeError:
        ok = True
    assert ok
    print("OK failure path")

if __name__ == "__main__":
    test_model_forward(); test_thresholds_success(); test_thresholds_failure()
    print("All smoke tests passed.")

# ============================================================
# STAGE 2: auto_config — thresholds ell_1, ell_2 via water-filling
# ============================================================
import json, numpy as np
from scipy.optimize import brentq


def calculate_thresholds(rho_0, alpha, ssm_state=64, nca_k=3,
                         c_S=0.05, c_N=0.2, c_A=1.0, max_len=2048):
    def rho(L): return rho_0 * (L ** (-alpha))
    def eq_ssm(L): return c_S * np.exp(-L / ssm_state) - rho(L)
    def eq_attn(L): return rho(L) - c_A
    try:
        l1 = brentq(eq_ssm, 1, 1000)
    except ValueError:
        l1 = 1.0
    try:
        l2 = brentq(eq_attn, 1, max_len)
    except ValueError:
        l2 = max_len / 2.0
    if l1 > 4 * ssm_state:
        ssm_state = int(np.ceil(l1 / 2))
    n_ssm = min(4, max(1, int(np.ceil(l1 / ssm_state))))
    gap = l2 - l1
    nca_T = min(50, max(3, int(np.ceil(gap / nca_k))))
    n_nca = min(2, max(1, int(np.ceil(gap / (nca_T * nca_k + 1e-9)))))
    n_attn = 1 if (max_len - l2) < 100 else 2
    print(f"ell_1 = {l1:.2f}, ell_2 = {l2:.2f}, ssm_state = {ssm_state}")
    print(f"Layers: SSM={n_ssm}, NCA={n_nca} (T={nca_T}), Attn={n_attn}")
    if gap > 100:
        print(f"WARNING: gap = {gap:.1f} > 100. NCA may not fully cover the region.")
    return {
        "ell_1": int(np.ceil(l1)), "ell_2": int(np.ceil(l2)),
        "ssm_state": ssm_state, "nca_T": nca_T, "nca_k": nca_k,
        "n_layers_per_region": (n_ssm, n_nca, n_attn),
    }


with open("rho_result.json") as f:
    rho = json.load(f)

print(f"Loaded: alpha={rho['alpha']:.4f}, residual={rho['max_rel_residual']*100:.2f}%")

if rho["max_rel_residual"] > 0.05:
    print("STOP: power law fails. HWF-LM not applicable.")
else:
    config = calculate_thresholds(rho["rho_0"], rho["alpha"], max_len=1024)
    with open("hwf_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nConfig saved to hwf_config.json")
    print(json.dumps(config, indent=2))

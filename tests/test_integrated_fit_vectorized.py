"""curve_fit passes arrays of W; integrated_power_law must accept them."""
import sys
import types
import numpy as np
from scipy.optimize import curve_fit

try:
    import tiktoken  # noqa: F401
    import datasets  # noqa: F401
except ImportError:
    stub = types.ModuleType("hwf_lm.data")
    stub.make_dataloader = None
    stub.load_tinystories_tokens = None
    sys.modules["hwf_lm.data"] = stub

from hwf_lm.measure.rho import integrated_power_law


def test_scalar():
    y = integrated_power_law(4, 1.0, 1.5, 5.0)
    assert isinstance(y, float)
    assert y < 5.0


def test_array():
    W = np.array([1.0, 2.0, 4.0, 8.0])
    y = integrated_power_law(W, 1.0, 1.5, 5.0)
    assert isinstance(y, np.ndarray)
    assert y.shape == (4,)
    assert np.all(np.diff(y) <= 1e-12)


def test_curve_fit_call():
    H0 = 5.0
    W = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    true_r, true_a = 0.8, 1.4
    y = np.array([integrated_power_law(float(w), true_r, true_a, H0) for w in W])
    y = y + np.random.default_rng(0).normal(0, 1e-4, size=len(W))
    popt, _ = curve_fit(
        lambda w, r, a: integrated_power_law(w, r, a, H0),
        W, y, p0=[0.5, 1.3], bounds=([1e-6, 1.05], [10, 2.5]), maxfev=5000,
    )
    assert popt[0] > 0 and 1.05 <= popt[1] <= 2.5
    print("OK vectorized fit", popt)


if __name__ == "__main__":
    test_scalar()
    test_array()
    test_curve_fit_call()
    print("All integrated_fit vectorized tests passed.")

def __getattr__(name):
    if name == "measure_rho":
        from .rho import measure_rho
        return measure_rho
    if name == "calculate_thresholds":
        from .thresholds import calculate_thresholds
        return calculate_thresholds
    if name == "measure_all_costs":
        from .costs import measure_all_costs
        return measure_all_costs
    raise AttributeError(name)
__all__ = ["measure_rho", "calculate_thresholds", "measure_all_costs"]

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
from scipy.optimize import minimize


def _positive_part(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def solve_nonlinear_optimization(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nonlinear factory optimization.

    Cost per machine i:
        base: c_i * x_i
        penalty: k_i * (x_i - t_i)_+^2

    Constraints:
        - Demand: sum(p_i * x_i) >= D
        - Bounds: 0 <= x_i <= h_i_max

    Parameters

    input_data : dict
        Expected keys:
        - "demand": float
        - "machines": list of {
            "name": str,
            "max_hours_per_day": float,
            "production_rate": float,
            "cost_per_hour": float
          }
        - "nonlinear_params": {machine_name: {
                "threshold_hours": float,
                "penalty_coeff": float
            }, ...}

    Returns

    dict
        {
            "status": str,
            "total_cost": float | None,
            "total_production": float | None,
            "machine_hours": {machine_name: float},
            "raw_status": str,  # from scipy OptimizeResult.message
        }
    """
    demand = float(input_data["demand"])
    machines = input_data["machines"]
    nonlinear_params = input_data.get("nonlinear_params", {})

    n = len(machines)
    names: List[str] = [m["name"] for m in machines]

    c = np.array([float(m["cost_per_hour"]) for m in machines])
    p = np.array([float(m["production_rate"]) for m in machines])
    h_max = np.array([float(m["max_hours_per_day"]) for m in machines])

    # For machines without explicit nonlinear params, use threshold = max_hours (no penalty until then)
    # and penalty_coeff = 0.0 (no penalty at all).
    thresholds = np.zeros(n)
    k = np.zeros(n)
    for i, name in enumerate(names):
        params = nonlinear_params.get(name, {})
        thresholds[i] = float(params.get("threshold_hours", h_max[i]))
        k[i] = float(params.get("penalty_coeff", 0.0))

    def objective(x: np.ndarray) -> float:
        # x: hours per machine
        base_cost = c * x
        penalty_term = k * _positive_part(x - thresholds) ** 2
        return float(np.sum(base_cost + penalty_term))

    # Constraint: sum(p_i * x_i) - D >= 0
    cons = [
        {
            "type": "ineq",
            "fun": lambda x, p=p, demand=demand: np.dot(p, x) - demand,
        }
    ]

    # Bounds: 0 <= x_i <= h_i_max
    bounds = [(0.0, hi) for hi in h_max]

    # Initial guess: half of max hours
    x0 = 0.5 * h_max

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
    )

    raw_status = result.message
    success = bool(result.success)

    if not success:
        # Even if SLSQP fails, we return something reasonable
        return {
            "status": "Failure",
            "raw_status": raw_status,
            "total_cost": None,
            "total_production": None,
            "machine_hours": {},
        }

    x_opt = np.clip(result.x, 0.0, h_max)
    machine_hours = {name: float(x_opt[i]) for i, name in enumerate(names)}

    total_production = float(np.dot(p, x_opt))
    total_cost = float(objective(x_opt))

    status = "Optimal" if success else "Failure"

    return {
        "status": status,
        "raw_status": raw_status,
        "total_cost": total_cost,
        "total_production": total_production,
        "machine_hours": machine_hours,
    }

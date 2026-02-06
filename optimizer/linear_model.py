from __future__ import annotations

from typing import Dict, Any

import pulp


def solve_linear_optimization(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Linear factory optimization problem.

    input_data : dict
        Parsed JSON with at least:
        - "demand": float
        - "machines": list of {
            "name": str,
            "max_hours_per_day": float,
            "production_rate": float,
            "cost_per_hour": float
          }


    dict
        {
            "status": str,
            "total_cost": float | None,
            "total_production": float | None,
            "machine_hours": {machine_name: float},
            "raw_status": str,  # solver-specific status string
        }
    """
    demand = float(input_data["demand"])
    machines = input_data["machines"]

    # Create LP problem
    prob = pulp.LpProblem("Factory_Optimization_Linear", pulp.LpMinimize)

    # Decision variables: hours each machine runs
    hours_vars: Dict[str, pulp.LpVariable] = {}
    for m in machines:
        name = m["name"]
        max_hours = float(m["max_hours_per_day"])
        var = pulp.LpVariable(f"hours_{name}", lowBound=0, upBound=max_hours)
        hours_vars[name] = var

    # Minimize total cost
    prob += pulp.lpSum(
        m["cost_per_hour"] * hours_vars[m["name"]] for m in machines
    ), "Total_Operating_Cost"

    # Demand constraint: sum(p_i * x_i) >= D
    prob += (
        pulp.lpSum(
            m["production_rate"] * hours_vars[m["name"]] for m in machines
        )
        >= demand,
        "Demand_Constraint",
    )

    prob.solve()  # uses default solver

    raw_status = pulp.LpStatus[prob.status]
    status: str
    if raw_status == "Optimal":
        status = "Optimal"
    elif raw_status in {"Infeasible", "Unbounded"}:
        status = "Infeasible"
    else:
        status = raw_status

    # If infeasible or not optimal, we still return info but mark cost/production as None
    if status != "Optimal":
        return {
            "status": status,
            "raw_status": raw_status,
            "total_cost": None,
            "total_production": None,
            "machine_hours": {},
        }

    # Extract solution
    machine_hours: Dict[str, float] = {
        name: float(var.value()) for name, var in hours_vars.items()
    }

    total_cost = sum(
        m["cost_per_hour"] * machine_hours[m["name"]] for m in machines
    )
    total_production = sum(
        m["production_rate"] * machine_hours[m["name"]] for m in machines
    )

    return {
        "status": status,
        "raw_status": raw_status,
        "total_cost": total_cost,
        "total_production": total_production,
        "machine_hours": machine_hours,
    }

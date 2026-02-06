from optimizer.linear_model import solve_linear_optimization


def test_linear_feasible_simple_optimal():
    # Two machines:
    # Machine A: cost per unit = 0.2
    # Machine B: cost per unit = 0.15
    # Demand = 100 units
    data = {
        "demand": 100,
        "machines": [
            {
                "name": "A",
                "max_hours_per_day": 10,
                "production_rate": 10,
                "cost_per_hour": 2,
            },
            {
                "name": "B",
                "max_hours_per_day": 10,
                "production_rate": 20,
                "cost_per_hour": 3,
            },
        ],
    }

    result = solve_linear_optimization(data)

    assert result["status"] == "Optimal"
    assert result["total_production"] >= data["demand"] - 1e-6

    hours = result["machine_hours"]
    # B should be 5 hours, A should be 0.
    assert abs(hours["B"] - 5.0) < 1e-4
    assert abs(hours["A"]) < 1e-4

    # Cost approximately 5 * 3 = 15
    assert abs(result["total_cost"] - 15.0) < 1e-4


def test_linear_infeasible_demand():
    # Not enough capacity to meet demand:each machine can run at most 1 hour
    data = {
        "demand": 1000,
        "machines": [
            {
                "name": "A",
                "max_hours_per_day": 1,
                "production_rate": 10,
                "cost_per_hour": 2,
            },
            {
                "name": "B",
                "max_hours_per_day": 1,
                "production_rate": 20,
                "cost_per_hour": 3,
            },
        ],
    }

    result = solve_linear_optimization(data)

    # Expect infeasible or similar
    assert result["status"] in {"Infeasible", "Undefined", "Not Solved"}
    assert result["total_cost"] is None
    assert result["total_production"] is None
    assert result["machine_hours"] == {}

from optimizer.nonlinear_model import solve_nonlinear_optimization


def test_nonlinear_basic_scenario():
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
        "nonlinear_params": {
            "A": {"threshold_hours": 8.0, "penalty_coeff": 2.0},
            "B": {"threshold_hours": 5.0, "penalty_coeff": 1.0},
        },
    }

    result = solve_nonlinear_optimization(data)

    assert result["status"] == "Optimal"
    assert result["total_production"] >= data["demand"] - 1e-6

    hours = result["machine_hours"]
    assert set(hours.keys()) == {"A", "B"}

    # Check bounds
    assert 0.0 <= hours["A"] <= data["machines"][0]["max_hours_per_day"]
    assert 0.0 <= hours["B"] <= data["machines"][1]["max_hours_per_day"]


def test_nonlinear_handles_missing_params():
    # Only one machine has nonlinear params; the other defaults to no penalty.
    data = {
        "demand": 50,
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
                "production_rate": 10,
                "cost_per_hour": 2,
            },
        ],
        "nonlinear_params": {
            "A": {"threshold_hours": 4.0, "penalty_coeff": 2.0}
            # No entry for B
        },
    }

    result = solve_nonlinear_optimization(data)

    assert result["status"] == "Optimal"
    assert result["total_production"] >= data["demand"] - 1e-6

    hours = result["machine_hours"]
    assert set(hours.keys()) == {"A", "B"}

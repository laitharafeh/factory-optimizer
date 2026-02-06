from __future__ import annotations

import argparse
from typing import Literal

from optimizer.utils import load_input_data
from optimizer import (
    solve_linear_optimization,
    solve_nonlinear_optimization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Factory optimization engine (linear & nonlinear)."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "nonlinear"],
        required=True,
        help="Which model to run: 'linear' or 'nonlinear'.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON input file.",
    )
    return parser.parse_args()


def print_results(model: Literal["linear", "nonlinear"], results: dict, demand: float):
    print(f"Model: {model}")
    print(f"Status: {results.get('status')}")
    raw_status = results.get("raw_status")
    if raw_status and raw_status != results.get("status"):
        print(f"Solver status (raw): {raw_status}")
    print()

    total_production = results.get("total_production")
    total_cost = results.get("total_cost")

    print(f"Demand: {demand:.2f} units")

    if total_production is not None:
        print(f"Total units produced: {total_production:.2f}")
    else:
        print("Total units produced: N/A")

    if total_cost is not None:
        if model == "nonlinear":
            print(f"Total (base cost + penalty): {total_cost:.2f}")
        else:
            print(f"Total cost: {total_cost:.2f}")
    else:
        print("Total cost: N/A")

    print("\nMachine schedules:")
    machine_hours = results.get("machine_hours", {})
    if not machine_hours:
        print("  (no schedule, problem may be infeasible or solver failed)")
    else:
        for name, hours in machine_hours.items():
            print(f"- {name}: {hours:.2f} hours")


def main():
    args = parse_args()
    data = load_input_data(args.input)
    model = args.model
    demand = float(data["demand"])

    if model == "linear":
        results = solve_linear_optimization(data)
    elif model == "nonlinear":
        results = solve_nonlinear_optimization(data)
    else:
        raise ValueError(f"Unknown model: {model}")

    print_results(model, results, demand)


if __name__ == "__main__":
    main()

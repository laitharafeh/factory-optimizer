# Factory Optimizer

A small Python optimization engine that computes optimal operating schedules
for machines in a factory using:

- Linear Programming via pulp
- Nonlinear Optimization via scipy.optimize.minimize

The goal is to demonstrate practical optimization modeling (LP + nonlinear)

## Project Structure

```
factory-optimizer/
  ├─ README.md
  ├─ requirements.txt
  ├─ data/
  │    └─ example_input.json
  ├─ optimizer/
  │    ├─ __init__.py
  │    ├─ linear_model.py
  │    ├─ nonlinear_model.py
  │    └─ utils.py
  ├─ main.py
  └─ tests/
       ├─ test_linear_model.py
       └─ test_nonlinear_model.py

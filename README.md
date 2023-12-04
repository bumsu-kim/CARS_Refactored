# CARS: Curvature-Aware Random Search

## Installation
Clone this repo and run 
```bash
pip install -e .
```
from the root directory of the repo to install `CARS` with editable mode.

## Usage
There is a sample script (`benchmark.py`) and a notebook (`carstest.ipynb`).

In these samples you can find the basic usage of this package.

Here is a quick guide:

1. `import` the `CARS` and other optimizers from `cars.code.optimizers.coptimizers`
2. Define your functions to optimize (black-box) in `cars/code/problems/`. For instance, define `my_func` in `cars/code/problems/my_functions.py`
    ```python
    def my_func(x: ndarray) -> float :
        # some computation ...
        return f_value
    ```
3. Configure the optimizer in `configs/`. For instance, in `my_config.json`,
    ```json
    "CARS_simple":{
        "Otype": "CARS",
        "description": "config for CARS optimizer",
        "f_module": "cars.code.problems.my_functions",
        "f_name": "my_func",
        "randgen": "uniform_sphere",
        "h": 0.1,
        "Lhat": 1.5
    }
    ```

4. Create an optimizer object with your `x0` and run `optimize()`:
    ```python
    from cars.code.utils.util_funcs import setup_optimizer
    opt = setup_optimizer(config, x0 = x0)
    opt.optimize()
    ```

Then you're all set!
# CARS: Curvature-Aware Random Search

## Installation
Clone this repo and run 
```bash
pip install -e .
```
from the root directory of the repo to install `CARS` with editable mode.

## Minimal Example
Here is a minimal working example for a quick preview:
```python
import numpy as np
from cars.code.utils.setup_utils import setup_default_optimizer


def my_func(x: np.ndarray) -> float:
    """simple quadratic function"""
    return np.sum(x**2)


def main():
    dim = 30
    x0 = 3 * np.random.randn(dim) + 1.0
    opt = setup_default_optimizer("CARS-CR", f=my_func, x0=x0)
    opt.optimize()


if __name__ == "__main__":
    main()
```

and a sample result:
```
Initialization done.
        Config: Default config for CARS-CR optimizer
        func name: convex_quartic       max evaluation = 3000
        f(x_0) = 272.4834569252656
        budget = 3000
        f_target = 0.2724834569252656

Finished: Reached the function target
Current status = Reached the function target
        eval_cnt = 889
        fsol = 2.650881e-01
```

## Usage
1. Define the problem to solve (_i.e._ your function to minimize) in `cars/code/problems/`.  
   _e.g._ in `cars/code/problems/my_functions.py`,
   ```python
    def my_func(x: ndarray) -> float :
        # some computation ...
        return f_value
    ```
2. Set your optimizer's configuration in a separate `json` file in `cars/configs/`.  
   This makes it easy to record and manage the results.  
   _e.g._ in `cars/config/my_config.json`,
   ```json
    {
    "CARS_simple":{
        "Otype": "CARS",
        "description": "config for CARS optimizer",
        "f_module": "cars.code.problems.my_functions",
        "f_name": "my_func",
        "randgen": "uniform_sphere",
        "h": 0.1,
        "Lhat": 1.5
    }
    }
    ```
3. Read the configuration and set your optimizer with this
   and your starting point `x0`.  
   Call `optimize()` to start optimization.  

   _e.g._ in `cars/executables/my_test.py`,  
   ```python
   from cars.code.utils.util_funcs import read_configs_from_json
   from cars.code.utils.setup_utils import setup_optimizer

   config = read_configs_from_json("cars/configs/my_test.json")["CARS_simple"]
   opt = setup_optimizer(config, x0=x0)
   opt.optimize()
   ```


There is a sample script `benchmark.py` for comparing various algorithms implemented in this project.
You can find more usage there (_e.g._ using custom `call_back` functions)

## Sample run: `benchmark.py`
Run `benchmark.py` to compare the `CARS`, `CARS-CR`, `CARS-NQ`, and `Nesterov-Spokoiny` optimizers for the 30-dimensional Rosenbrock function:
```bash
~/CARS_Refactored$ python cars/executables/benchmark.py 
```
Sample result:
```
------------
Testing CARS_simple
Initialization done.
        Config: config for CARS optimizer
        func name: rosenbrock   max evaluation = 50000
        f(x_0) = 225553.67240992823
        budget = 50000
        f_target = 1e-08

Finished: Reached the max number of evaluations
Current status = Reached the max number of evaluations
        eval_cnt = 50000
        fsol = 8.569526e+01


[    0.   240.   258. 16168.]

------------
Testing CARS_CR
Initialization done.
        Config: config for CARS-CR optimizer
        func name: rosenbrock   max evaluation = 50000
        f(x_0) = 225553.67240992823
        budget = 50000
        f_target = 1e-08

Finished: Reached the max number of evaluations
Current status = Reached the max number of evaluations
        eval_cnt = 50000
        fsol = 7.205167e+01


[    0.     0.     0. 12499.     0.]

------------
Testing CARS_NQ
Initialization done.
        Config: config for CARS-NQ optimizer
        func name: rosenbrock   max evaluation = 50000
        f(x_0) = 225553.67240992823
        budget = 50000
        f_target = 1e-08

Finished: Reached the max number of evaluations
Current status = Reached the max number of evaluations
        eval_cnt = 50000
        fsol = 7.925918e+01


[8333.    0.    0.    0.    0.    0.    0.]

------------
Testing Nesterov-Spokoiny
Initialization done.
        Config: config for Nesterov-Spokoiny optimizer. Rosenbrock function requires large L
        func name: rosenbrock   max evaluation = 50000
        f(x_0) = 225553.67240992823
        budget = 50000
        f_target = 1e-08

Finished: Reached the max number of evaluations
Current status = Reached the max number of evaluations
        eval_cnt = 50000
        fsol = 1.301358e+02


[   0. 3831. 3920. 8915.]

------------
```
The lists of integers at the ends are the `safeguard_counter`, which counts how often each candidate is chosen at the safeguard step (see, _e.g._, Line 9 of Algorithm 1.)
# CARS: Curvature-Aware Random Search

## Installation
Clone this repo and run 
```bash
pip install -e .
```
from the root directory of the repo to install `CARS` with editable mode.

## Minimal Example
Here is a [minimal working example](cars/executables/minimal_ex.py) for a quick preview:
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
        Description: Default config for CARS-CR optimizer
        func name: convex_quartic       budget = 3000
        f(x_0) = 204.0644816576005
        budget = 3000
        f_target = 0.20406448165760052

Finished: Reached the function target
Current status = Reached the function target
        eval_cnt = 805
        fsol = 1.906433e-01
```
When setting up your optimizer with `setup_default_optimizer`, the [default configuration](cars/configs/default.json) is used.  
If a finer tuning of your optimizer is required (e.g., sampling radius, (relative) smoothness parameter, etc.), see the next section.

Another minimal example can be found in this [notebook](cars/executables/minimal_ex2.ipynb) as well.

## Fine-tuing Optimizers using `config.json`
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
    When benchmarking many algorithms with common settings, _e.g._, same function, same budget, etc., use `"Common"` object in the json. See [benchmark_rosenbrock.json](cars/configs/benchmark_rosenbrock.json) for an illustration.  

3. Read the configuration file and set your optimizer with it.
   Also set your starting point `x0`.  
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

## `benchmark.py`: An Example for Comparing Multiple Optimizers
Run [`benchmark.py`](cars/executables/benchmark.py) to compare the `CARS`, `CARS-CR`, `CARS-NQ`, and `Nesterov-Spokoiny` optimizers for the 30-dimensional Rosenbrock function:
```bash
> ~/CARS_Refactored$ python cars/executables/benchmark.py 
```
The configuration file used for this test is [benchmark_rosenbrock.json](cars/configs/benchmark_rosenbrock.json).  

### Sample result:
```
------------
Testing CARS_simple
Initialization done.
        Description: config for CARS optimizer
        func name: rosenbrock   budget = 50000
        f(x_0) = 860240.2425351156
        budget = 50000
        f_target = 1e-08

Finished: Reached the max number of evaluations
Current status = Reached the max number of evaluations
        eval_cnt = 50000
        fsol = 1.205601e+01

Safeguard counter: [    0.   312.   277. 16077.]

------------
Testing CARS_CR
Initialization done.
        Description: config for CARS-CR optimizer
        func name: rosenbrock   budget = 50000
        f(x_0) = 860240.2425351156
        budget = 50000
        f_target = 1e-08

Finished: Reached the max number of evaluations
Current status = Reached the max number of evaluations
        eval_cnt = 50000
        fsol = 1.978933e+01

Safeguard counter: [    0.     0.     0. 12499.     0.]


... (omitted) ...


------------
Testing Nesterov-Spokoiny
Initialization done.
        Description: config for Nesterov-Spokoiny optimizer. Rosenbrock function requires large L
        func name: rosenbrock   budget = 50000
        f(x_0) = 860240.2425351156
        budget = 50000
        f_target = 1e-08

Finished: Reached the max number of evaluations
Current status = Reached the max number of evaluations
        eval_cnt = 50000
        fsol = 8.017286e+01

Safeguard counter: [    0.  2768.  2798. 11100.]
```
The lists of integers at the ends are the `safeguard_counter`, which counts how often each candidate is chosen at the safeguard step (see, _e.g._, Line 9 of Algorithm 1.)

`benchmark.py` also plots and saves the results into files in `cars/figures/` directory.
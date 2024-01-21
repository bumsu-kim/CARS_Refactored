import numpy as np
import matplotlib.pyplot as plt
from cars.code.base.base_optimizers import BaseOptimizer
from cars.code.utils.util_funcs import read_configs_from_json, plot_res
from cars.code.utils.setup_utils import setup_optimizer
import os  # for os.path.exits() for checking the figures folder


def callback(optimizer: BaseOptimizer):
    """Callback function for recording hu and gu.
    If the field hu_history/gu_history doesn't exist, first create it.
    Then start recording.
    Args:
        optimizer (BaseOptimizer): optimizer to apply this callback
    """
    if hasattr(optimizer, "hu"):
        if not hasattr(optimizer, "hu_history"):
            optimizer.hu_history = []
        optimizer.hu_history.append(optimizer.hu)
    if not hasattr(optimizer, "gu_history"):
        optimizer.gu_history = []
    optimizer.gu_history.append(np.abs(optimizer.gu))


def main():
    """Benchmark a set of algorithms defined in `benchmark_rosenbrock.json` for the Rosenbrock function.

    The common configuratoin (e.g., total budget) can be adjusted in the `Common` object in the json file.
    (Note: if `budget_dim_ratio` is positive, `budget` is ignored.)

    The benchmark function can also be changed in `Common` area. (e.g., `convex_quartic`)

    Control optimizer-specific settings (e.g., step size) in each optimizer's field.


    """
    configs = read_configs_from_json("cars/configs/benchmark_rosenbrock.json")
    opts = {}
    # Set problem dimension and the same initial point
    dim = 30
    x0 = 3 * np.random.randn(dim) + 1.0
    for config_name, config in configs.items():
        print(f"\n------------\nTesting {config_name}")
        opts[config_name] = setup_optimizer(config, x0=x0, call_back=callback)
        opts[config_name].optimize()
        print(f"Safeguard counter: {opts[config_name].safeguard_counter}")

    ### Plot Results
    # Check directory
    save_dir = "cars/figures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # plot f_history
    plot_res(
        opts,
        os.path.join(save_dir, "benchmark_f.png"),
        "f_history",
    )
    # plot gu_history
    plot_res(
        opts,
        os.path.join(save_dir, "benchmark_gu.png"),
        "gu_history",
        alpha=0.2,
        grid=None,
    )
    # plot hu_history
    plot_res(
        opts,
        os.path.join(save_dir, "benchmark_hu.png"),
        "hu_history",
        alpha=0.2,
    )


if __name__ == "__main__":
    main()

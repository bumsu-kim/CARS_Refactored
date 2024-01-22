import numpy as np
from cars.utils import read_configs_from_json, plot_res, setup_optimizer
import os  # for os.path.exits() for checking the figures folder


def callback(optimizer):
    """Callback function for recording hu and gu.

    If the field hu_history/gu_history doesn't exist, first create it.
    Then start recording.

    Args:
        optimizer: optimizer to apply this callback
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
    configs = read_configs_from_json("configs/benchmark_rosenbrock.json")
    opts = {}  # will contain optimizer objects
    # Set problem dimension and the same initial point
    dim = 30
    x0 = 3 * np.random.randn(dim) + 1.0

    # Set up optimizers and run
    for config_name, config in configs.items():
        print(f"\n------------\nTesting {config_name}")
        opts[config_name] = setup_optimizer(config, x0=x0, call_back=callback)
        opts[config_name].optimize()
        print(f"Safeguard counter: {opts[config_name].safeguard_counter}")

    ### Plot Results
    # Check directory
    save_dir = "figures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # plot f_history
    plot_res(
        opts,
        "f_history",
        os.path.join(save_dir, "benchmark_f.png"),
    )
    # plot gu_history
    plot_res(
        opts,
        "gu_history",
        os.path.join(save_dir, "benchmark_gu.png"),
        alpha=0.3,
        grid=None,
        xlabel="Iterations",
    )
    # plot hu_history
    plot_res(
        opts,
        "hu_history",
        os.path.join(save_dir, "benchmark_hu.png"),
        alpha=0.3,
        xlabel="Iterations",
    )


if __name__ == "__main__":
    main()

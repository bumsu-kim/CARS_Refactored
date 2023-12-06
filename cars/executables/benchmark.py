import numpy as np
import matplotlib.pyplot as plt
from cars.code.base.base_optimizers import BaseOptimizer
from cars.code.utils.util_funcs import read_configs_from_json
from cars.code.utils.setup_utils import setup_optimizer


def callback(optimizer: BaseOptimizer):
    """Callback function for recording hu and gu"""
    if hasattr(optimizer, "hu"):
        if not hasattr(optimizer, "hu_history"):
            optimizer.hu_history = []
        optimizer.hu_history.append(optimizer.hu)
    if not hasattr(optimizer, "gu_history"):
        optimizer.gu_history = []
    optimizer.gu_history.append(np.abs(optimizer.gu))


def main():
    configs = read_configs_from_json("cars/configs/benchmark_rosenbrock.json")
    opts = {}
    dim = 30
    x0 = 3 * np.random.randn(dim) + 1.0
    for config_name, config in configs.items():
        print(f"\n------------\nTesting {config_name}")
        opts[config_name] = setup_optimizer(config, x0=x0, call_back=callback)
        opts[config_name].optimize()
        print(opts[config_name].safeguard_counter)

    # plot f_history
    plt.figure()
    for config_name in configs:
        plt.plot(opts[config_name].f_history, label=config_name)
        # set y logscale
        plt.yscale("log")
        plt.legend()
        plt.grid(which="both")
        plt.savefig("cars/figures/benchmark.png")

    # plot hu_history (size of d^2f / du^2)
    plt.figure()
    for config_name in configs:
        if hasattr(opts[config_name], "hu_history"):
            plt.plot(opts[config_name].hu_history, label=config_name, alpha=0.2)
        # set y logscale
        plt.yscale("log")
        plt.legend()
        plt.grid(which="both")
        plt.savefig("cars/figures/benchmark_hu.png")

    # plot gu_history (size of df / du)
    plt.figure()
    for config_name in configs:
        plt.plot(opts[config_name].gu_history, label=config_name, alpha=0.2)
        # set y logscale
        plt.yscale("log")
        plt.legend()
        plt.grid(which="both")
        plt.savefig("cars/figures/benchmark_gu.png")


if __name__ == "__main__":
    main()

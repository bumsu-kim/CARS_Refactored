import numpy as np
import matplotlib.pyplot as plt
from cars.code.utils.util_funcs import read_configs_from_json
from cars.code.utils.setup_utils import setup_optimizer


def callback(optimizer):
    if hasattr(optimizer, "hu"):
        if not hasattr(optimizer, "hu_history"):
            optimizer.hu_history = []
        optimizer.hu_history.append(optimizer.hu)
    if not hasattr(optimizer, "gu_history"):
        optimizer.gu_history = []
    optimizer.gu_history.append(optimizer.gu)

def main():
    configs = read_configs_from_json("cars/configs/benchmarks.json")
    opts = {}
    dim = 30
    x0 = np.random.randn(dim) + 1.0
    for config_name, config in configs.items():
        print(f"Testing {config_name}")
        opts[config_name] = setup_optimizer(config, x0 = x0, call_back = callback)
        opts[config_name].optimize()
        print(opts[config_name].CARS_counter)
    
    # plot f_history
    plt.figure()
    for config_name in configs:
        # plot opts[config_name].f_history with yscale = log (hold on)
        plt.plot(opts[config_name].f_history, label = config_name)
        # set y logscale
        plt.yscale("log")
        plt.legend()
        plt.savefig("cars/figures/benchmark.png")

    # plot hu_history (size of d^2f / du^2)
    plt.figure()
    for config_name in configs:
        # plot opts[config_name].f_history with yscale = log (hold on)
        if hasattr(opts[config_name], "hu_history"):
            plt.plot(opts[config_name].hu_history, label = config_name, alpha = 0.2)
        # set y logscale
        plt.yscale("log")
        plt.legend()
        plt.savefig("cars/figures/benchmark_hu.png")

    # plot gu_history (size of df / du)
    plt.figure()
    for config_name in configs:
        # plot opts[config_name].f_history with yscale = log (hold on)
        plt.plot(opts[config_name].gu_history, label = config_name, alpha = 0.2)
        # set y logscale
        plt.yscale("log")
        plt.legend()
        plt.savefig("cars/figures/benchmark_gu.png")


if __name__ == "__main__":
    main()
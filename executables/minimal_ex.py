import numpy as np
from cars.utils import setup_default_optimizer


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

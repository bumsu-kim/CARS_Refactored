from cars.utils.util_funcs import read_configs_from_json
import importlib
import inspect
from cars.optimizers.base_optimizers import BaseOptimizer


def load_optimizers() -> dict:
    """Load optimizers defined in `cars/optimizers/optimizers.py`"""
    optimizers_module = importlib.import_module("cars.optimizers.optimizers")
    optimizers_dict = {}
    for name, obj in inspect.getmembers(optimizers_module, inspect.isclass):
        if obj.__module__ == "cars.optimizers.optimizers":
            optimizers_dict[getattr(obj, "Otype", None)] = obj
    return optimizers_dict


optimizers_dict = {}  # Initialized when setup_optimizer is called


def setup_optimizer(
    config: dict[str, dict[str, float] | float | str], **kwargs
) -> BaseOptimizer:
    """Setup an optimizer from a configuration

    Args:
        config (dict[str, dict[str, float] | float | str]): dictionary of configurations.

    Returns:
        callable: optimizer
    """
    global optimizers_dict
    if config["Otype"] not in optimizers_dict:
        optimizers_dict = load_optimizers()
    return optimizers_dict[config["Otype"]](config, **kwargs)


def setup_default_optimizer(config_name: str, **kwargs) -> BaseOptimizer:
    """Setup a default optimizer (read from `configs/default.json`)

    Args:
        config_name (str): name of the configuration in `default.json`

    Returns:
        callable: optimizer
    """
    if not hasattr(setup_default_optimizer, "configs"):
        setup_default_optimizer.configs = read_configs_from_json("configs/default.json")
    config = setup_default_optimizer.configs[config_name]
    return setup_optimizer(config, **kwargs)

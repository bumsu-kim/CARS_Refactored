from cars.code.utils.util_funcs import read_configs_from_json

import importlib
import inspect


def load_optimizers():
    """Load optimizers defined in `cars/code/optimizers/optimizers.py`"""
    optimizers_module = importlib.import_module("cars.code.optimizers.optimizers")
    optimizers_dict = {}
    for name, obj in inspect.getmembers(optimizers_module, inspect.isclass):
        if obj.__module__ == "cars.code.optimizers.optimizers":
            optimizers_dict[getattr(obj, "Otype", None)] = obj
    return optimizers_dict


optimizers_dict = (
    load_optimizers()
)  # Read all optimizers from cars/code/optimizers/optimizers.py


def setup_optimizer(
    config: dict[str, dict[str, float] | float | str], **kwargs
) -> callable:
    """Setup an optimizer

    Args:
        config (dict[str, dict[str, float] | float | str]): configuration

    Returns:
        callable: optimizer
    """
    return optimizers_dict[config["Otype"]](config, **kwargs)


def setup_default_optimizer(config_name: str, **kwargs) -> callable:
    """Setup a default optimizer (read from `cars/configs/default.json`)

    Args:
        config_name (str): name of the configuration in `default.json`

    Returns:
        callable: optimizer
    """
    if not hasattr(setup_default_optimizer, "configs"):
        setup_default_optimizer.configs = read_configs_from_json(
            "cars/configs/default.json"
        )
    config = setup_default_optimizer.configs[config_name]
    return setup_optimizer(config, **kwargs)

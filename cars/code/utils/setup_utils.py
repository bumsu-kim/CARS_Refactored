from cars.code.optimizers.optimizers import CARS, CARSNQ, CARSCR, Nesterov


def setup_optimizer(config: dict[str, dict[str, float] | float | str], **kwargs) -> callable:
    """Setup an optimizer

    Args:
        config (dict[str, dict[str, float] | float | str]): configuration

    Returns:
        callable: optimizer
    """
    Otype = config["Otype"]
    if Otype == "CARS":
        optimizer = CARS(config, **kwargs)
    elif Otype == "CARS-NQ":
        optimizer = CARSNQ(config, **kwargs)
    elif Otype == "CARS-CR":
        optimizer = CARSCR(config, **kwargs)
    elif Otype == "Nesterov":
        optimizer = Nesterov(config, **kwargs)

    return optimizer

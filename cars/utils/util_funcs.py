import numpy as np
import numpy.matlib
import json
import logging
import matplotlib.pyplot as plt
from scipy.special import roots_hermite


def read_configs_from_json(json_file: str) -> dict[str, dict[str, float] | float | str]:
    """read configs from a json file

    Args:
        json_file (str): path to the json file

    Returns:
        dict[str, dict[str, float] | float | str]: configs
    """
    with open(json_file, "r") as f:
        configs = json.load(f)
    if "Common" in configs:
        common_config = configs["Common"]
        configs.pop("Common")
    else:
        common_config = {}
    for config in configs.values():
        for key in common_config:
            if key not in config:
                config[key] = common_config[key]

    return configs


## Helper functions for post-processing
def plot_res(
    opts: dict,
    field_name: str,
    save_file_name: str | None = None,
    **kwargs,
):
    """Plot the results and save it in a file.

    Args:
        opts (BaseOptimizer|dict[BaseOptimizer]): A single optimizer, or a dictionary of optimizers
        save_file_name (str): file name to store the result
        field_name (str): fields to plot. For instance, f_history, hu_history or gu_history.

    Keyword Args:
        Arguments for figure control or plt.plot(). For example:

        [Figure Control]
        - title (str): plot title
        - figsize (tuple(int,int)): figure size
        - yscale (str): y-scale of the plot, "log" is default. Use "linear" for linear plots.
        - legend: Specify to "None" to suppress showing. Default is to show legend
        - grid: Specify to "None" to suppress showing. Default is to show both
        - xlabel (str|None): show xlabel as specified. Default is "Evaluations". Specify to "None" to suppress showing
        - ylabel (str|None): show ylabel as specified. Default is "field_name". Specify to "None" to suppress showing

        [Plot Attributes] (anything not in [Figure Control])
        - alpha, linestyle, etc.
    """
    if not isinstance(opts, dict):  # if single object
        opts = {getattr(opts, "OType", "Default_Label"): opts}

    # First check if the field_name is valid
    isValid = any(
        hasattr(opt, field_name) for opt in opts.values()
    )  # valid if any of them has it
    if not isValid:
        raise ValueError(f'None of the optimizers have the attribute "{field_name}".')
    plt.figure(figsize=kwargs.get("figsize", (12, 9)))
    figure_attributes = [
        "title",
        "figsize",
        "yscale",
        "legend",
        "grid",
        "xlabel",
        "ylabel",
    ]
    kwargs_plt = {k: v for k, v in kwargs.items() if k not in figure_attributes}
    for label, opt in opts.items():
        if hasattr(opt, field_name):
            plt.plot(getattr(opt, field_name), label=label, **kwargs_plt)
    if "title" in kwargs:
        plt.title(kwargs.get("title"))
    if "yscale" in kwargs:
        plt.yscale(kwargs.get("yscale", "log"))
    else:
        plt.yscale("log")
    if kwargs.get("legend", "default_is_not_None") is not None:
        plt.legend()
    if kwargs.get("grid", "default_is_not_None") is not None:
        plt.grid(which=kwargs.get("grid", "both"))
    if kwargs.get("xlabel", "default_is_not_None") is not None:
        plt.xlabel(kwargs.get("xlabel", "Evaluations"))
    if kwargs.get("ylabel", "default_is_not_None") is not None:
        plt.ylabel(kwargs.get("ylabel", field_name))
    # Save to figure
    if save_file_name is not None:
        if opt.verbose > 0:
            print(f"Saving plots for {field_name}.")
        plt.savefig(save_file_name)


## Helper functions for Computing derivative-related quantites
def central_difference(fp: float, fm: float, h: float) -> float:
    """Compute the central difference

    Args:
        fp (float): f(x0 + h)
        fm (float): f(x0 - h)
        h (float): finite difference step size

    Returns:
        float: central difference
    """
    return (fp - fm) / 2.0 / h


def forward_difference(f0: float, fp: float, h: float) -> float:
    """Compute the forward difference

    Args:
        fp (float): f(x0 + h)
        f0 (float): f(x0)
        h (float): finite difference step size

    Returns:
        float: forward difference
    """
    return (fp - f0) / h


def central_difference_both(
    f0: float, fp: float, fm: float, h: float
) -> tuple[float, float]:
    """Compute the first and second order central differences

    Args:
        fp (float): f(x0 + h)
        fm (float): f(x0 - h)
        f0 (float): f(x0)
        h (float): finite difference step size

    Returns:
        tuple[float, float]: first and second central difference gradient
    """
    return (fp - fm) / 2.0 / h, (fp - 2.0 * f0 + fm) / h**2


def get_directional_derivs_nq(
    fs: np.ndarray, h: float, gh_weight: np.ndarray, gh_value: np.ndarray
) -> tuple[float, float, float, float]:
    """Approximate the first, second, third, and fourth order directional derivative using numerical quadrature

    Args:
        fs (np.ndarray): function values
        h (float): finite difference step size
        gh_weight (np.ndarray): Gauss-Hermite weights
        gh_value (np.ndarray): Gauss-Hermite quadratures

    Returns:
        tuple[float, float, float, float]: first, second, third, and fourth order numerical quadrature
    """
    fsgh = fs * gh_weight
    gh_value = np.transpose(gh_value)
    grad_u = 1.0 / np.sqrt(np.pi) / h * np.sum(fsgh * (np.sqrt(2.0) * gh_value))
    hess_u = 1.0 / np.sqrt(np.pi) / h**2 * np.sum(fsgh * (2 * gh_value**2 - 1))
    D3f_u = (
        1.0
        / np.sqrt(np.pi)
        / h**3
        * np.sum(fsgh * (np.sqrt(8.0) * gh_value**3 - 3.0 * np.sqrt(2.0) * gh_value))
    )
    D4f_u = (
        1.0
        / np.sqrt(np.pi)
        / h**4
        * np.sum(fsgh * (4 * gh_value**4 - 6 * 2 * gh_value**2 + 3))
    )
    return grad_u, hess_u, D3f_u, D4f_u


## Helper functions for random sampling
def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """Normalize a matrix

    Args:
        mat (ndarray): vector or matrix, or tensor

    Returns:
        ndarray: normalized matrix
    """
    mat /= np.sqrt(np.sum(np.square(mat)))
    return mat


def normalize_matrices(mat: np.ndarray) -> np.ndarray:
    """Normalize matrices along the first dimension

    Args:
        mat (np.ndarray): matrices of shape (n_samples, *shape)

    Returns:
        np.ndarray: normalized matrices
    """
    # Calculate the squared sum along the last dimensions (keeping the first dimension intact)
    norms = np.sqrt(np.sum(np.square(mat), axis=tuple(range(1, mat.ndim))))

    # Reshape norms for broadcasting
    reshaped_norms = norms.reshape(-1, *([1] * (mat.ndim - 1)))

    # Normalize each matrix
    normalized_matrices = mat / reshaped_norms

    return normalized_matrices


def dim2coord(dim: int, shape: tuple[int]) -> tuple[int]:
    """Convert a 1D coordinate to a multi-dimensional coordinate

    Args:
        dim (int): 1D coordinate
        shape (tuple[int]): shape of the multi-dimensional coordinate

    Returns:
        tuple[int]: multi-dimensional coordinate
    """
    coord = []
    for i in range(len(shape)):
        coord.append(dim % shape[i])
        dim = dim // shape[i]
    return tuple(coord)

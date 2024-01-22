import numpy as np
from pyparsing import Any
from cars.utils.util_funcs import normalize_matrices, normalize_matrix, dim2coord


def uniform_sphere(n_samples: int, shape: tuple[int], params: dict[str, Any] = None):
    """Generate uniformly distributed samples on a sphere

    Args:
        n_samples (int): number of samples
        shape (tuple[int]): shape of each sample (can be 1D, 2D, or more than 2D)
        params (dict[str, float], optional): "radius" and "center". Defaults to None.

    Returns:
        ndarray: generated samples of shape (n_samples, *shape) if n_samples>1, or (*shape) if n_samples=1
    """
    mat = gaussian(n_samples, shape, params=None)  # standard normal
    mat = (
        normalize_matrices(mat) if n_samples > 1 else normalize_matrix(mat)
    )  # normalize each sample
    if params is None:
        return mat
    else:
        return params["radius"] * mat + params["center"]


def gaussian(n_samples: int, shape: tuple[int], params: dict[str, Any] = None):
    """Generate Gaussian samples

    Args:
        n_samples (int): number of samples
        shape (tuple[int]): shape of each sample (can be 1D, 2D, or more than 2D)
        params (dict[str, float], optional): "mean" and "std". Defaults to None.

    Returns:
        ndarray: generated samples of shape (n_samples, *shape) if n_samples>1, or [*shape] matrix of samples if n_samples=1
    """
    if params is None:
        params = {"mean": 0.0, "std": 1.0}
    if n_samples > 1:
        mat = np.random.normal(
            loc=params["mean"], scale=params["std"], size=(n_samples, *shape)
        )
    else:
        mat = np.random.normal(loc=params["mean"], scale=params["std"], size=shape)
    return mat


def random_coordinates(
    n_samples: int, shape: tuple[int], params: dict[str, Any] = None
):
    """Generate random coordinate vectors

    Args:
        n_samples (int): number of samples
        shape (tuple[int]): shape of each sample (can be 1D, 2D, or more than 2D)
        params (dict[str, Any], optional): "p": probabilities of each coordinate to be chosen. Defaults to None.
    """
    # param can give the probability of choosing each coordinate
    dim = np.prod(shape)
    if params is None:
        params["p"] = None  # default (uniform)
    coords = np.random.choice(dim, size=n_samples, replace=True, p=params["p"])

    mat = np.zeros((n_samples, *shape))
    for i in range(n_samples):
        mat[(i,) + dim2coord(coords[i])] = 1.0

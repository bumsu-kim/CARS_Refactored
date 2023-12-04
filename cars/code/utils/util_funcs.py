import numpy as np
import importlib
import json
import logging


def load_function(module_name: str, function_name: str) -> callable:
    """Load a function from a module

    Args:
        module_name (str): name of the module
        function_name (str): name of the function

    Returns:
        callable: function handle
    """
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def read_configs_from_json(json_file: str) -> dict[str, dict[str, float] | float | str]:
    """read configs from a json file

    Args:
        json_file (str): path to the json file

    Returns:
        dict[str, dict[str, float] | float | str]: configs
    """
    with open(json_file, "r") as f:
        configs = json.load(f)
    common_config = configs.get("Common", {})
    configs.pop("Common")
    for config in configs.values():
        for key in common_config:
            if key not in config:
                config[key] = common_config[key]

    return configs



def setup_logger(logger_name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger

    Args:
        logger_name (str): name of the logger
        log_file (str): path to the log file
        level (int, optional): logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: logger
    """
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fileHandler = logging.FileHandler(log_file, mode="w")
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger

## Helper functions for Computing Finite Difference
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


# class Xinfo:
#     """Class to store the status of the optimizer"""

#     def __init__(self, x0: np.ndarray, budget: int, dim: int = None):
#         self.x0 = x0
#         self.dim = len(x0) if dim is None else dim  # use numpy method instead?
#         init_sz = budget  # if maxEval <= dim*1000 else dim*1000
#         self.x = np.zeros((init_sz, self.dim))
#         self.i = 0  # current pointer loc
#         self.exceed_max_evals = False
#         self.max_i = budget
#         self.shape = self.x0.shape
#         self.put_x(x0)

#     # subscript operator
#     def __getitem__(self, j: int) -> np.ndarray:
#         """Returns the j-th x

#         Args:
#             j (int): index of x

#         Raises:
#             ValueError: if j exceeds the max number of evaluations

#         Returns:
#             np.ndarray: x
#         """
#         if j < self.max_i:
#             return self.x[j, :]
#         else:
#             raise ValueError(f"{j} exceeds the max num evals {self.max_i}")

#     def put_x(self, x: np.ndarray):
#         if not self.exceed_max_evals:
#             self.x[self.i, :] = x
#             self.i += 1
#         if self.i >= self.max_i:
#             self.exceed_max_evals = True

#     def newest(self):
#         return self.bracket(self.i - 1)


# class fVals:
#     """Class to store the status of the optimizer"""

#     def __init__(self, f0: float, maxEval: int = 1000):
#         self.rec = True
#         self.vals = np.zeros(maxEval)
#         self.i = 0  # current pointer loc
#         self.best_val: float = np.inf
#         self.best_i = self.i
#         self.max_i = maxEval
#         self.exceed_max_evals = False
#         self.put_val(f0)

#     def put_val(self, fx: float, threshold=0.0):
#         if not self.exceed_max_evals:
#             self.vals[self.i] = fx
#             if fx < self.best_val - threshold:  # better by this amount
#                 self.best_val = fx
#                 self.best_i = self.i
#             self.i += 1

#         if self.i >= self.max_i:
#             self.exceed_max_evals = True

#     def __getitem__(self, j: int):  # later change the name
#         if j < self.max_i:
#             return self.vals[j]
#         else:
#             raise ValueError(f"{j} exceeds the max num evals {self.max_i}")

#     def newest(self):
#         return self.vals[self.i - 1]

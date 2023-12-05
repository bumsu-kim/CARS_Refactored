import numpy as np
import numpy.typing as npt
from pyparsing import Any
from cars.code.utils.util_funcs import load_function
from cars.code.utils.util_cls import Status, ReachedMaxEvals


class BaseOptimizer:
    Otype = "Base"

    def __init__(
        self,
        config: dict[str, Any],
        x0: np.ndarray = None,
        f: callable = None,
        call_back: callable = None,
    ):
        """Generates a BaseOptim object
        Otype: "Base"
        Args:
            x0 (np.ndarray): initial x
            f (callable): function to optimize (if not given, read from config)
            config (dict[str, Any]): configuration, maybe containing the followings:
                description (str): description of the optimizer/parameter
                x0 (np.ndarray): initial x (if not given in the constructor)
                dim (int): dimension of the problem (if not given in the constructor)
                f_module (str): module name of the function to optimize
                f_name (str): function name of the function to optimize
                record_x_history (bool): whether to record x history (default: True)
                budget (int): max number of function evaluations
                f_target (float): target f value
                verbose (int): verbosity level
        """
        self.description = config.get("description", "Default BaseOptim")

        if x0 is None:
            self.x0 = config.get("x0", np.zeros(1))
        else:
            self.x0 = x0
        self.dim = config.get("dim", len(self.x0))

        if "budget_dim_ratio" in config:
            if config["budget_dim_ratio"] > 0:
                self.budget = config["budget_dim_ratio"] * self.dim
            else:
                self.budget = config.get("budget", 1000)
        else:
            self.budget = config.get("budget", 1000)

        self.xshape = self.x0.shape

        self.sol = self.x0  # current solution

        self.record_x_history = config.get("record_x_history", True)
        if self.record_x_history:
            self.x_history = [self.sol]
        else:
            self.x_history = None

        if f is None:
            self.f = load_function(config["f_module"], config["f_name"])
        else:
            self.f = f

        self.fname = config["f_name"]
        self.fname_long = config["f_module"] + "." + config["f_name"]

        self.eval_cnt = 0
        self.f_history = np.zeros(self.budget)
        self.f_history[0] = self.f(self.sol)
        self.fsol = self.f_history[0]  # current f value
        self.eval_cnt += 1

        self.f_best = self.f_history[0]
        self.x_best = x0

        if (
            config.get("target_accuracy", 0) > 0
        ):  # if "target_accuracy" exists and positive
            self.f_target = (
                config.get("f_target", 0)
                + (self.f_history[0] - config.get("f_target", 0))
                * config["target_accuracy"]
            )
        else:
            self.f_target = config.get("f_target", -np.inf)

        self.verbose = config.get("verbose", 1)

        self.status = Status()
        self.call_back = call_back if call_back is not None else lambda x: None
        if self.verbose > 0:
            print("Initialization done.")
            if self.verbose > 1:
                print(
                    f"\tConfig: {self.description}"
                    + f"\n\tfunc name: {self.fname}"
                    + f"\tmax evaluation = {self.budget}\n"
                    + f"\tf(x_0) = {self.f(self.sol)}\n"
                    + f"\tbudget = {self.budget}\n"
                    + f"\tf_target = {self.f_target}\n"
                )
            if self.verbose > 2:
                print(f"\tInitial x = {self.sol}")

    def feval(self, x: np.ndarray, isCounted=True) -> float:
        if isCounted:
            fx = self.f(x)
            self.record(x, fx)  # also increment the counter
            return fx
        else:
            # eval_cnt does not increase, and do not record
            fx = self.f(x)
            return fx

    def fevals(self, X: list[np.ndarray], isCounted=True) -> list[float]:
        fxs = []
        for x in X:
            fx = self.feval(x, isCounted)
            fxs.append(fx)
            if self.status.finished:
                break
        return fxs

    def record(self, x: np.ndarray, fx: float):
        if self.record_x_history:
            self.x_history.append(x)
        self.f_history[self.eval_cnt] = fx
        self.update_if_best(x, fx)
        self.eval_cnt += 1
        self.status.check_reached_ftn_target(self.f_target, self.f_best)
        self.status.check_reached_max_evals(self.eval_cnt, self.budget)
        if self.status.reached_max_evals:
            raise ReachedMaxEvals

    def update_if_best(self, x: np.ndarray, fx: float):
        # update if a new best is found
        # can be overridden with threshold, etc.
        if fx < self.f_best:
            self.f_best = fx
            self.x_best = x

    def curr(self) -> tuple[np.ndarray, float]:
        return self.sol, self.fsol

    def best(self) -> tuple[np.ndarray, float]:
        return self.x_best, self.f_best

    def optimize(self):
        try:
            while self.status.finished is False:
                self.step()
                self.call_back(self)
        except ReachedMaxEvals:
            self.sol = self.x_best
            self.fsol = self.f_best

        self.opt_finished()

    def step(self):
        """Must be overridden"""
        pass

    def opt_finished(self):
        self.f_history = self.f_history[: self.eval_cnt]
        if self.verbose > 0:
            print(f"Finished: {self.status}")
            if self.verbose > 1:
                self.print_status()

    def print_status(self):
        print(f"Current status = {self.status}")
        print(f"\teval_cnt = {self.eval_cnt}")
        print(f"\tfsol = {self.fsol:.6e}")
        if self.verbose > 2:
            print(f"\tsol = {self.sol}")


class StochasticOptimizer(BaseOptimizer):
    """Line search optimizer
    BaseOptim: Base class for other optimizer classes
    """

    OType = "Stochastic"

    def __init__(
        self,
        config: dict[str, Any],
        x0: np.ndarray = None,
        f: callable = None,
        call_back: callable = None,
    ):
        """Generates a StochasticOptimizer object
        OType: "Stochastic"
        Args:
            x0 (np.ndarray): initial x
            f (callable): function to optimize (if not given, read from config)
            config (dict[str, Any]): configuration, maybe containing the followings:
                description (str): description of the optimizer/parameter
                budget (int): max number of function evaluations
                x0 (np.ndarray): initial x (if not given in the constructor)
                dim (int): dimension of the problem (if not given in the constructor)
                record_x_history (bool): whether to record x history (default: True)
                f_module (str): module name of the function to optimize
                f_name (str): function name of the function to optimize
                f_target (float): target f value
                alpha (float): step size
                verbose (int): verbosity level
        """
        super().__init__(config, x0, f, call_back)
        self.randgen = load_function(
            "cars.code.utils.random_funcs", config.get("randgen", "gaussian")
        )

    def get_random_samples(self, n_samples=1):
        return self.randgen(n_samples, self.xshape)

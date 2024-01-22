# `from cars.optimizers import *` imports the following:
__all__ = ["CARS", "CARSCR", "CARSNQ", "Nesterov", "STP"]

from .base_optimizers import StochasticOptimizer
from cars.utils.util_funcs import (
    central_difference_both,
    central_difference,
    get_directional_derivs_nq,
)
import numpy as np
from pyparsing import Any
from scipy.special import roots_hermite


class CARS(StochasticOptimizer):
    Otype = "CARS"

    def __init__(
        self,
        config: dict[str, Any],
        x0: np.ndarray = None,
        f: callable = None,
        call_back: callable = None,
    ):
        super().__init__(config, x0, f, call_back)
        self.h = config.get("h", 1e-2)
        self.Lhat = config.get("Lhat", 1.5)
        self.safeguard_counter = np.zeros(4)  # record how the next step is determined,
        # 0: f(x_k), 1: f(x_k+h*u), 2: f(x_k-h*u), 3: f(x_cars)

    def step(self):
        u = self.get_random_samples()
        x, fx = self.curr()
        h = self.h / np.sqrt(1 + self.eval_cnt)  # Use decreasing sampling radius
        x_p_hu = x + h * u
        x_m_hu = x - h * u
        fx_p_hu = self.feval(x_p_hu)
        fx_m_hu = self.feval(x_m_hu)
        self.gu, self.hu = central_difference_both(fx, fx_p_hu, fx_m_hu, h)
        self.delta_x_cars = -self.gu / np.abs(self.hu) / self.Lhat * u
        x_cars = x + self.delta_x_cars
        fx_cars = self.feval(x_cars)
        f_candidates = [fx, fx_p_hu, fx_m_hu, fx_cars]
        # update x by choosing the best candidate
        i_best = np.argmin(f_candidates)
        self.sol = [x, x_p_hu, x_m_hu, x_cars][i_best]
        self.fsol = f_candidates[i_best]
        self.safeguard_counter[i_best] += 1


class CARSCR(CARS):
    Otype = "CARS-CR"

    def __init__(
        self,
        config: dict[str, Any],
        x0: np.ndarray = None,
        f: callable = None,
        call_back: callable = None,
    ):
        super().__init__(config, x0, f, call_back)
        self.M = config.get("M", 10)
        self.safeguard_counter = np.zeros(5)  # record how the next step is determined,
        # 0: f(x_k), 1: f(x_k+h*u), 2: f(x_k-h*u),
        # 3: f(x_cars_p), 4: f(x_cars_m)

    def step(self):
        u = self.get_random_samples()
        x, fx = self.curr()
        h = self.h / np.sqrt(1 + self.eval_cnt)
        x_p_hu = x + h * u
        x_m_hu = x - h * u
        fx_p_hu = self.feval(x_p_hu)
        fx_m_hu = self.feval(x_m_hu)
        self.gu, self.hu = central_difference_both(fx, fx_p_hu, fx_m_hu, h)
        Lhat = 0.5 + np.sqrt(0.25 + self.M * np.abs(self.gu) / 2.0 / (self.hu**2))
        self.delta_x_cars_cr = -self.gu / self.hu / Lhat * u
        x_cars_cr_p = x + self.delta_x_cars_cr
        fx_cars_cr_p = self.feval(x_cars_cr_p)
        x_cars_cr_m = x - self.delta_x_cars_cr
        fx_cars_cr_m = self.feval(x_cars_cr_m)
        f_candidates = [fx, fx_p_hu, fx_m_hu, fx_cars_cr_p, fx_cars_cr_m]
        # update x by choosing the best candidate
        i_best = np.argmin(f_candidates)
        self.sol = [x, x_p_hu, x_m_hu, x_cars_cr_p, x_cars_cr_m][i_best]
        self.fsol = f_candidates[i_best]
        self.safeguard_counter[i_best] += 1


class CARSNQ(CARS):
    Otype = "CARS-NQ"

    def __init__(
        self,
        config: dict[str, Any],
        x0: np.ndarray = None,
        f: callable = None,
        call_back: callable = None,
    ):
        super().__init__(config, x0, f, call_back)
        self.N_nq = config.get("nq", 5)
        gh = roots_hermite(self.N_nq)
        self.gh_value = np.expand_dims(gh[0], axis=1)
        self.gh_weight = gh[1]
        if self.N_nq % 2 == 0:
            self.safeguard_counter = np.zeros(
                self.N_nq + 3
            )  # record how the next step is determined,
            # 0: f(x_CARS+), 1: f(x_CARS-), 2: f(x_k),
            # 3, ... : f(x_k + h * num_quad * u)
        else:
            self.safeguard_counter = np.zeros(
                self.N_nq + 2
            )  # record how the next step is determined,
            # 0: f(x_CARS+), 1: f(x_CARS-),
            # 2, ... : f(x_k + h * num_quad * u)
            # In this case f(x_k) is at 2 + (self.N_nq-1)/2 -th position
            # e.g. N_nq = 5, f(x_k) is at 4-th position

    def get_quad_points(self, x: np.ndarray, h: float, u: np.ndarray) -> np.ndarray:
        """Get the quadrature points

        Args:
            x (np.ndarray): current point
            h (float): finite difference step size
            u (np.ndarray): random direction
        Returns:
            np.ndarray: quadrature points (to evaluate function values)
        """
        xs = (
            np.matlib.repmat(x, self.N_nq, 1) + h * np.sqrt(2.0) * self.gh_value * u
        )  # N_nq x dim
        return xs

    def eval_at_quad_pts(self, xs, fx=None):
        if self.N_nq % 2 == 0:
            fs = self.fevals(xs)  # self.N_nq evaluations total
        else:  # can reuse fval = f(x)
            # self.N_nq - 1 evaluations total
            xm = xs[: (self.N_nq - 1) // 2, :]
            xp = xs[-(self.N_nq - 1) // 2 :, :]
            fs = np.empty(self.N_nq)
            fs[: (self.N_nq - 1) // 2] = self.fevals(xm)
            fs[(self.N_nq - 1) // 2] = fx
            fs[-(self.N_nq - 1) // 2 :] = self.fevals(xp)
        return fs

    def step(self):
        u = self.get_random_samples()
        x, fx = self.curr()
        h = self.h / np.sqrt(1 + self.eval_cnt)
        xs = self.get_quad_points(x, h, u)
        fs = self.eval_at_quad_pts(xs, fx)
        self.gu, self.hu, self.d3f_u, self.d4f_u = get_directional_derivs_nq(
            fs, h, self.gh_weight, self.gh_value
        )
        d3sup = np.abs(self.d3f_u) + np.abs(
            self.gu / self.hu * self.d4f_u
        )  # estimate of sup|f'''| near x
        # 1/Lhat estimated from higher order derivatives
        Lhat = 0.5 + np.sqrt(0.25 + np.abs(self.gu * d3sup / self.hu**2 / 3))
        # see proposition 5.3

        self.delta_x_cars_nq = -self.gu / self.hu / Lhat * u
        x_cars_nq_p = x + self.delta_x_cars_nq
        fx_cars_nq_p = self.feval(x_cars_nq_p)
        x_cars_nq_m = x - self.delta_x_cars_nq
        fx_cars_nq_m = self.feval(x_cars_nq_m)
        if self.N_nq % 2 == 0:
            f_candidates = [fx_cars_nq_p, fx_cars_nq_m, fx, *fs]
        else:
            f_candidates = [fx_cars_nq_p, fx_cars_nq_m, *fs]
        # update x by choosing the best candidate
        i_best = np.argmin(f_candidates)
        if self.N_nq % 2 == 0:
            self.sol = [x_cars_nq_p, x_cars_nq_m, x, *xs][i_best]
        else:
            self.sol = [x_cars_nq_p, x_cars_nq_m, *xs][i_best]
        self.fsol = f_candidates[i_best]
        self.safeguard_counter[i_best] += 1


class Nesterov(StochasticOptimizer):
    Otype = "Nesterov"

    def __init__(
        self,
        config: dict[str, Any],
        x0: np.ndarray = None,
        f: callable = None,
        call_back: callable = None,
    ):
        super().__init__(config, x0, f, call_back)
        self.h = config.get("h", 1e-2)
        self.L = config.get("L", 40 * (4 + self.dim))  # doesn't converge if too small
        self.safeguard_counter = np.zeros(4)  # record how the next step is determined,
        # 0: f(x_k), 1: f(x_k+h*u), 2: f(x_k-h*u), 3: f(x_ns)

    def step(self):
        u = self.get_random_samples()
        x, fx = self.curr()
        h = self.h / np.sqrt(1 + self.eval_cnt)
        x_p_hu = x + h * u
        x_m_hu = x - h * u
        fx_p_hu = self.feval(x_p_hu)
        fx_m_hu = self.feval(x_m_hu)
        self.gu = central_difference(fx_p_hu, fx_m_hu, h)
        self.delta_x = -self.gu / self.L * u
        x_ns = x + self.delta_x
        fx_ns = self.feval(x_ns)
        f_candidates = [fx, fx_p_hu, fx_m_hu, fx_ns]
        # update x by choosing the best candidate
        i_best = np.argmin(f_candidates)
        self.sol = [x, x_p_hu, x_m_hu, x_ns][i_best]
        self.fsol = f_candidates[i_best]
        # self.sol = x_ns
        # self.fsol = fx_ns
        self.safeguard_counter[i_best] += 1


class STP(StochasticOptimizer):
    Otype = "STP"

    def __init__(
        self,
        config: dict[str, Any],
        x0: np.ndarray = None,
        f: callable = None,
        call_back: callable = None,
    ):
        super().__init__(config, x0, f, call_back)
        self.h = config.get("h", 1e-2)
        self.L = config.get("L", 40 * (4 + self.dim))

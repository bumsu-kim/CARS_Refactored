from sympy import Line
from cars.code.base.base_optimizers import LineSearchOptimizer
from cars.code.utils.util_funcs import central_difference_both, central_difference
import numpy as np
from pyparsing import Any


class CARS(LineSearchOptimizer):
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
        self.CARS_counter = np.zeros(
            4
        )  # record how the next step is determined,
        # 0: f(x_k), 1: f(x_k+h*u), 2: f(x_k-h*u), 3: f(x_cars)

    def step(self):
        u = self.get_random_samples()
        x, fx = self.curr()
        h = self.h / np.sqrt(1 + self.eval_cnt)
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
        self.CARS_counter[i_best] += 1

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
        self.CARS_counter = np.zeros(
            5
        )  # record how the next step is determined,
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
        Lhat = 0.5 + np.sqrt(0.25 + self.M*np.abs(self.gu)/2.0/(self.hu**2))
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
        self.CARS_counter[i_best] += 1
    


class CARSNQ(CARS):
    Otype = "CARS-NQ"
    def __init__(
        self,
        config: dict[str, Any],
        x0: np.ndarray = None,
        f: callable = None,
    ):
        super().__init__(config, x0, f)
        self.N_nq = config.get("nq", 5)
        self.nq = []  # numerical quadratures
        self.nqrange = np.arange(1, self.N_nq // 2 + 1)  # positive half [1, 2, ..]

    def eval_essential(self, u):
        u = np.random.randn(self.X.shape)
        x, fx = self.curr()

        x_p_hu = [x + self.nq[i] * self.h * u for i in self.nqrange]
        x_m_hu = [x - self.nq[i] * self.h * u for i in self.nqrange]

        fx_p_hu = self.fevals(x_p_hu)
        fx_m_hu = self.fevals(x_m_hu)
        

class Nesterov(LineSearchOptimizer):
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
        self.L = config.get("L", 40*(4+self.dim))
        self.CARS_counter = np.zeros(
            4
        ) # record how the next step is determined,
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
        # self.sol = [x, x_p_hu, x_m_hu, x_ns][i_best]
        # self.fsol = f_candidates[i_best]
        self.sol = x_ns
        self.fsol = fx_ns
        self.CARS_counter[i_best] += 1
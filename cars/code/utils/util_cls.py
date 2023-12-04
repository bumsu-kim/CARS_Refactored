class Status:
    """Class to store the status of the optimizer"""

    def __init__(self):
        self.initialized = False
        self.reached_ftn_target = False
        self.reached_max_evals = False
        self.finished = False
        self.message = "Not finished yet"

    def __str__(self):
        return self.message

    def check_reached_ftn_target(self, f_target: float, f_best: float):
        if f_best <= f_target:
            self.reached_ftn_target = True
            self.finished = True
            self.message = "Reached the function target"

    def check_reached_max_evals(self, eval_cnt: int, budget: int):
        if eval_cnt >= budget:
            self.reached_max_evals = True
            self.finished = True
            self.message = "Reached the max number of evaluations"


class ReachedMaxEvals(Exception):
    """Exception to be raised when the optimizer is terminated"""

    pass

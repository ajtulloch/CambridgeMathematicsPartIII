import numpy as np
from collections import namedtuple

import logging
logging.basicConfig(level=logging.DEBUG)


class BarrierFunction(object):
    def __init__(self, n):
        pass

    def f(self, x):
        raise NotImplementedError()

    def grad_f(self, x):
        raise NotImplementedError()

    def hessian_f(self, x):
        raise NotImplementedError()

    def theta(self):
        raise NotImplementedError()


class ConeBarrier(BarrierFunction):
    def __init__(self, n):
        """
        Arguments:
        - `n`:
        """
        self._n = n

    def grad_f(self, x):
        return -1.0 / x

    def hessian_f(self, x):
        diagonal = 1.0 / np.square(x)
        return np.diagflat(diagonal)

    def f(self, x):
        return -np.sum(np.log(x))

    def theta(self):
        return self._n


class TimeStepper(object):
    """
    """

    def __init__(self, barrier_function, rho, tau):
        self._barrier_function = barrier_function
        self._rho = rho
        self._tau = tau

    def next_tau(self):
        return self._tau

    def next_t(self, current_t):
        return (1.0 + self._rho / np.sqrt(self._barrier_function.theta())) \
            * current_t

    def gap_bound(self, current_t):
        return 2 * self._barrier_function.theta() / current_t


class Solver(object):
    SolutionState = namedtuple("SolutionState", ["x", "y", "t"])

    def __init__(self, A, b, c, barrier_function, time_stepper):
        self._A = A
        self._b = b
        self._c = c
        self._barrier_function = barrier_function
        self._time_stepper = time_stepper

    @staticmethod
    def solve(barrier_class,
              A,
              b,
              c,
              rho,
              tau,
              initial_state,
              tolerance):
        barrier_function = barrier_class(np.size(c))
        time_stepper = TimeStepper(barrier_function, rho, tau)
        solver = Solver(A, b, c, barrier_function, time_stepper)
        for step in solver._solve(initial_state, tolerance):
            yield step

    def _solve(self, initial_state, tolerance):
        """
        Yields the sequence of iterates and the primal-dual gap
        """
        current_gap = 10 ** 100 # Random high number
        current_state = initial_state
        while current_gap > tolerance:
            yield (current_state, current_gap)
            current_state = self._next(current_state)
            (x, y, t) = current_state
            current_gap = self._time_stepper.gap_bound(t)

    def _next(self, solution_state):
        delta_x, delta_y = self._delta(solution_state)
        (x_k, y_k, t_k) = solution_state
        
        return Solver.SolutionState(
            x=x_k + self._time_stepper.next_tau() * delta_x,
            y=x_k + self._time_stepper.next_tau() * delta_y,
            t=self._time_stepper.next_t(t_k))

    def _delta(self, solution_state):
        (x_k, y_k, t_k) = solution_state

        H = self._A.T * \
            self._barrier_function.hessian_f(self._A * x_k - self._b) * \
            self._A
        G = -1.0 * self._time_stepper.next_t(t_k) * self._c - \
            self._A.T * self._barrier_function.grad_f(self._A * x_k - self._b)

        delta_x = H.I * G

        K = self._barrier_function.grad_f(self._A * x_k - self._b) + \
            self._barrier_function.hessian_f(self._A * x_k - self._b) * \
            self._A * delta_x
        delta_y = -1.0 / self._time_stepper.next_t(t_k) * K - y_k
        return (delta_x, delta_y)

def example():
    g = Solver.solve(
        barrier_class=ConeBarrier,
        A=np.matrix('1 0; 0 1'),
        b=np.matrix('0; 0'),
        c=np.matrix('-1; -1'),
        rho=0.5 * 1.0 / 10,
        tau=1.0,
        initial_state=Solver.SolutionState(x=np.matrix('1; 1'), y=np.matrix('0; 0'), t=1.0),
        tolerance=1e-5)
    print list(g)

example()

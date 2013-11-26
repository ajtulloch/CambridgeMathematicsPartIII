import numpy as np
import scipy as sp
from collections import namedtuple


class PDHGSolver(object):
    Evaluators = namedtuple('Evaluators', ['A', 'g', 'h', 'g_conj', 'h_conj'])
    SolutionState = namedtuple('SolutionState', ['x', 'y'])
    Parameters = namedtuple('Parameters', ['sigma', 'tau', 'theta'])

    def __init__(self, evaluators, parameters):
        self._evaluators = evaluators
        self._parameters = parameters

    @staticmethod
    def solve(A, g, h, g_conj, h_conj, sigma, tau, theta, tolerance, x_0, y_0):
        """Main entry point for this solver.

        Solves \inf g(x) + h(Ax) using the modified PDHG with parameters
        (sigma, tau, theta) to tolerance with initial state (x_0, y_0)
        """
        solver = PDHGSolver(
            evaluators=PDHGSolver.Evaluators(
                A=A,
                g=g,
                h=h,
                g_conj=g_conj,
                h_conj=h_conj,
            ),
            parameters=PDHGSolver.Parameters(
                sigma=sigma,
                tau=tau,
                theta=theta,
            ))

        initial_state = PDHGSolver.SolutionState(x=x_0, y=x_0)
        for step in solver._solve(initial_state, tolerance):
            yield step

    def _solve(self, initial_state, tolerance):
        current_gap = 10 ** 100
        current_state = initial_state
        while current_gap > tolerance:
            yield (current_state, current_gap)
            current_state = self._next(current_state)
            (x_k, y_k) = current_state
            current_gap = self._primal(x_k) - self._dual(y_k)

    def _primal(self, x):
        return self._evaluators.g(x) + \
            self._evaluators.h(self._evaluators.A * x)

    def _dual(self, y):
        return -self._evaluators.g_conj(-self._evaluators.A.T * y) - \
            self._evaluators.h_conj(y)

    def _lagrangian(self, x, y):
        return self._evaluators.g(x) - self._evaluators.h_conj(y) + \
            np.dot(self._evaluators.A * x, y)

    def _backward_step(self, f, x, step_size):
        def objective(y):
            return 0.5 * np.linalg.norm(x - y) ** 2 + step_size * f(y)
        return sp.optimize.minimize(objective, x).x

    def _next(self, current_state):
        (x_current, y_current) = current_state
        # Backward step WRT x_k.
        y_next = self._backward_step(
            f=lambda y: -self.lagrangian(x_current, y),
            x=y_current,
            step_size=self._parameters.sigma)

        x_next = self._backward_step(
            f=lambda x: self.lagrangian(x, y_next),
            x=x_current,
            step_size=self._parameters.tau)

        x_next = x_next + self._parameters.theta * (x_next - x_current)

        return PDHGSolver.SolutionState(x=x_next, y=y_next)

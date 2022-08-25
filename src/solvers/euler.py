from .solver import Solver

class Euler(Solver):

    @staticmethod
    def step(f, state, t, h):

        state = state + h * f(state, t)
        return state

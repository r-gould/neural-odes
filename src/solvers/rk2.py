from .solver import Solver

class RK2(Solver):

    @staticmethod
    def step(f, state, t, h):
        
        k1 = h * f(state, t)
        k2 = h * f(state + k1/2, t + h/2)
        
        state = state + k2
        return state
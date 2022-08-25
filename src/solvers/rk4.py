from .solver import Solver

class RK4(Solver):

    @staticmethod
    def step(f, state, t, h):
        
        k1 = h * f(state, t)
        k2 = h * f(state + k1/2, t + h/2)
        k3 = h * f(state + k2/2, t + h/2)
        k4 = h * f(state + k3, t + h)

        state = state + k1/6 + k2/3 + k3/3 + k4/6
        return state
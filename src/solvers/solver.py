class Solver:
    
    def __init__(self, step_size):

        self.step_size = step_size

    def solve(self, state, f, t0, t1):
        
        h = self.step_size if t0 < t1 else -self.step_size
        n_steps = int(abs(t1 - t0) / self.step_size)
        t = t0
        for _ in range(n_steps):
            state = self.step(f, state, t, h)
            t = t + h

        return state
    
    @staticmethod
    def step(f, state, t, h):

        raise NotImplementedError()
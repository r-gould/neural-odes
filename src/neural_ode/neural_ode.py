import torch
import torch.nn as nn

from .ode_autograd import ODEAutograd

class NeuralODE(nn.Module):

    def __init__(self, dynamics, solver):

        super().__init__()
        self.dynamics = dynamics
        self.solver = solver

    def forward(self, z0):
        
        params = self.dynamics.get_params()
        z1 = ODEAutograd.apply(z0, self.dynamics, self.solver, params)
        return z1
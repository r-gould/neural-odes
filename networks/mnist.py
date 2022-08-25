import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.timeconv import TimeConv
from src.neural_ode import NeuralODE

class MNIST(nn.Module):

    def __init__(self, ode_channels, solver):

        super().__init__()

        dynamics = MNISTDynamics(ode_channels)

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, ode_channels, kernel_size=4, stride=2, padding=1),

            NeuralODE(dynamics, solver),

            nn.Conv2d(ode_channels, 32, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=2, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        
        return self.network(x)

class MNISTDynamics(nn.Module):

    def __init__(self, dim):
        
        super().__init__()

        self.time_conv1 = TimeConv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.time_conv2 = TimeConv(dim, dim, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, t):

        out = F.leaky_relu(self.time_conv1(x, t))
        return F.leaky_relu(self.time_conv2(out, t))

    def get_params(self):

        p = self.parameters()
        return torch.cat([torch.flatten(param) for param in p])
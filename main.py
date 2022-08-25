import torch
import torch.nn as nn

import src.solvers as solvers
import networks

from utils import load_mnist, plot_stats
from src.trainer import Trainer

def main(dataloaders, network, epochs, lr, device="cuda"):

    network = network.to(device)
    
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(network.parameters(), lr=lr)

    trainer = Trainer(dataloaders, loss, optim)
    train_stats, valid_stats = trainer.train(network, epochs, validate=True, 
                                             device=device)
    
    plot_stats(train_stats, valid_stats)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloaders = load_mnist(batch_size=128)
    solver = solvers.Euler(step_size=5e-3)
    network = networks.MNIST(ode_channels=64, solver=solver)

    main(dataloaders, network, epochs=10, lr=1e-3, device=device)
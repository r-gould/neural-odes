import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

def load_mnist(batch_size):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds = MNIST(root="data/", train=True, download=True, transform=transform)
    valid_ds = MNIST(root="data/", train=False, download=True, transform=transform)
    
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size, shuffle=True)

    return {
        "train": train_dl, 
        "valid" : valid_dl
    }

def plot_stats(train_stats, valid_stats):

    train_losses, train_accs = train_stats
    valid_losses, valid_accs = valid_stats

    plt.title("Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Valid")
    plt.legend(loc="upper right")
    plt.savefig("plots/losses.png")
    plt.show()

    plt.title("Accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(train_accs, label="Train")
    plt.plot(valid_accs, label="Valid")
    plt.legend(loc="upper left")
    plt.savefig("plots/accs.png")
    plt.show()

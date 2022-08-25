import torch
import numpy as np

from tqdm import tqdm

class Trainer:

    def __init__(self, dataloaders, loss, optim):
        
        self.dataloaders = dataloaders
        self.loss = loss
        self.optim = optim

    def train(self, model, epochs, validate=True, save_model=True,
              device="cuda"):

        train_dl = self.dataloaders.get("train")
        train_losses, train_accs = [], []
        valid_losses, valid_accs = [], []

        for epoch in range(1, epochs+1):
            print("Epoch:", epoch)
            epoch_losses, epoch_accs = [], []

            for images, target in tqdm(train_dl):

                images = images.to(device)
                target = target.to(device)

                logits = model(images)
                loss = self.loss(logits, target)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                acc = self.accuracy(logits, target)
                epoch_losses.append(loss.item())
                epoch_accs.append(acc)

            if save_model:
                print("Saving model...")
                torch.save(model.state_dict(), f"neural-odes/saved/model_{epoch}.pt")
                print("Model saved")

            avg_loss = np.mean(epoch_losses[-50:])
            avg_acc = np.mean(epoch_accs[-50:])
            print("Avg. train loss:", avg_loss)
            print("Avg. train acc:", avg_acc)

            train_losses.append(avg_loss)
            train_accs.append(avg_acc)
            epoch_losses, epoch_accs = [], []

            if validate:
                valid_dl = self.dataloaders.get("valid")
                valid_loss, valid_acc = self.test(model, valid_dl, device=device)
                print("Avg. valid loss:", valid_loss)
                print("Avg. valid acc:", valid_acc)

                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)

        return (train_losses, train_accs), (valid_losses, valid_accs)

    @torch.no_grad()
    def test(self, model, test_dl, device="cuda"):

        test_losses, test_accs = [], []

        for images, target in test_dl:

            images = images.to(device)
            target = target.to(device)

            logits = model(images)
            loss = self.loss(logits, target)

            acc = self.accuracy(logits, target)
            test_losses.append(loss.item())
            test_accs.append(acc)

        return np.mean(test_losses), np.mean(test_accs)

    @staticmethod
    def accuracy(logits, target):
        
        labels = torch.argmax(logits, dim=-1)
        batch_size = len(target)
        acc = (labels == target).sum() / batch_size
        return acc.cpu()
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_data, stack_data, stack_windowed_data
from dataset_loaders import create_dataloaders
from models import CNNModel

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed_all(seed)
else:
    device = 'cpu'

print('Using device:', device)


def train(model: torch.nn.Module, train_loader, val_loader, criterion, optimizer, num_epochs) -> dict[str, list]:
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    avg_val_acc = 0
    for epoch in tqdm(range(num_epochs)):
        train_losses, train_acc = [], []
        model.train()
        for padded_sequences, lengths, labels in train_loader:
            padded_sequences = padded_sequences.to(device)

            optimizer.zero_grad()
            outputs = model(padded_sequences.float().to(device))

            loss = criterion(outputs.view(-1, 4), labels.long().view(-1).to(device))
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            _, predicted = torch.max(outputs.view(-1, 4), 1)
            accuracy = accuracy_score(labels.view(-1).numpy(), predicted.cpu().detach().numpy())
            train_acc.append(accuracy)

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        avg_train_accuracy = np.mean(train_acc)
        history['train_acc'].append(avg_train_accuracy)

        model.eval()
        val_losses, val_acc = [], []

        with torch.no_grad():
            for padded_sequences, lengths, labels in val_loader:
                outputs = model(padded_sequences.float().to(device))

                loss = criterion(outputs.view(-1, 4), labels.long().view(-1).to(device))
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs.view(-1, 4), 1)
                accuracy = accuracy_score(labels.view(-1).numpy(), predicted.cpu().detach().numpy())
                val_acc.append(accuracy)

        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        avg_val_accuracy = np.mean(val_acc)
        history['val_acc'].append(avg_val_accuracy)

    return history


if __name__ == '__main__':
    data = load_data()
    sequences, targets = stack_windowed_data(data)
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(sequences, targets, batch_size=32,
                                                                              return_class_weights=True,
                                                                              verbose=False)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0097285,
                           weight_decay=7.254576818661595e-05)

    history = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    print('Model trained with final validation accuracy:', history['val_acc'][-1])

    # todo: add saving the model with model name and key parameters

    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    axs[0].plot(history['train_loss'], label='Train Loss')
    axs[0].plot(history['val_loss'], label='Validation Loss')
    axs[0].set_xlabel('Epoch #')
    axs[0].set_ylabel('Loss')

    axs[1].plot(history['train_acc'], label='Train Accuracy')
    axs[1].plot(history['val_acc'], label='Validation Accuracy')
    axs[1].set_xlabel('Epoch #')
    axs[1].set_ylabel('Accuracy')
    plt.tight_layout()
    plt.legend()
    plt.show()

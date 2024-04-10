import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import random
import matplotlib.pyplot as plt
from os.path import join

from utils import load_data, stack_data, stack_windowed_data
from dataset_loaders import create_dataloaders, create_cnn_dataloaders
from models import CNNModel, LSTMModel

seed = 7777777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed_all(seed)
else:
    device = 'cpu'

print('Using device:', device)



def train(model: torch.nn.Module, train_loader, val_loader, criterion, optimizer, num_epochs,
          model_name: str = 'cnn-model', device='cuda') -> dict[str, list]:
    best_val_loss = float('inf')
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    model.to(device)
    for epoch in range(num_epochs):
        train_losses, train_acc = [], []
        model.train()
        for sequences, labels in train_loader:

            optimizer.zero_grad()
            outputs = model(sequences.float().to(device))

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
            for sequences, labels in val_loader:
                outputs = model(sequences.float().to(device))

                loss = criterion(outputs.view(-1, 4), labels.long().view(-1).to(device))
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs.view(-1, 4), 1)
                accuracy = accuracy_score(labels.view(-1).numpy(), predicted.cpu().detach().numpy())
                val_acc.append(accuracy)

        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        avg_val_accuracy = np.mean(val_acc)
        history['val_acc'].append(avg_val_accuracy)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}]: Train Loss: {round(avg_train_loss, 3)}, '
                  f'Validation Loss: {round(avg_val_loss, 3)}, Val Accuracy: {round(avg_val_accuracy, 3)}')

        if avg_val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join('models', f'{model_name}.pt'))

            with open(os.path.join('models', 'history', f'{model_name}.json'), 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=4)

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
            plt.savefig(os.path.join('models', 'plots', f'{model_name}.png'))
            plt.close()

            print(f'* Epoch [{epoch}/{num_epochs}]: Train Loss: {round(avg_train_loss, 3)}, '
                  f'Validation Loss: {round(avg_val_loss, 3)}, Val Accuracy: {round(avg_val_accuracy, 3)}')

    print('Best epoch:', best_epoch, 'with validation loss:', best_val_loss)

    return history


if __name__ == '__main__':
    data = load_data(file_path=join('data', 'W_AUGMENTED_DATA.json'))
    sequences, targets = stack_windowed_data(data)
    train_loader, val_loader, test_loader, class_weights = create_cnn_dataloaders(sequences, targets, batch_size=64,
                                                                                  return_class_weights=True,
                                                                                  verbose=False)

    model = CNNModel(
        cnn_dropout=0.5,
        n_blocks=1,
        out_channels=32
    ).to(device)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.00005,
                           weight_decay=0.0005)

    model_name = 'cnn'
    history = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=250,
                    model_name=model_name, device=device)

    print('Model trained. Final validation accuracy:', history['val_acc'][-1])

    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    axs[0].plot(history['train_loss'], label='Train Loss')
    axs[0].plot(history['val_loss'], label='Validation Loss')
    axs[0].set_xlabel('Epoch #')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(history['train_acc'], label='Train Accuracy')
    axs[1].plot(history['val_acc'], label='Validation Accuracy')
    axs[1].set_xlabel('Epoch #')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join('models', 'plots', f'{model_name}.png'))
    plt.show()
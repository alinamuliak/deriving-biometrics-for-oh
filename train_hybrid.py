import argparse
import json
import os
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import random
import matplotlib.pyplot as plt
from os.path import join

from utils import load_data, stack_windowed_data, check_max
from dataset_loaders import create_dataloaders
from models import HybridModel

seed = 7777777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed_all(seed)
else:
    device = 'cpu'

def train(model: torch.nn.Module, train_loader, val_loader, criterion, optimizer, num_epochs,
          model_name: str = 'bilstm-model') -> dict[str, list]:
    best_val_loss = float('inf')
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        train_losses, train_acc = [], []
        model.train()
        for padded_sequences, lengths, labels in train_loader:
            padded_sequences = padded_sequences.to(device)

            optimizer.zero_grad()
            outputs = model(padded_sequences.float().to(device), lengths)

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
                outputs = model(padded_sequences.float().to(device), lengths)

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


def main(args):
    data = load_data(file_path=join('data', 'W_AUGMENTED_DATA.json'))
    sequences, targets = stack_windowed_data(data)
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(sequences, targets,
                                                                              batch_size=args.batch_size,
                                                                              return_class_weights=True,
                                                                              verbose=False)

    model = HybridModel(
        n_conv_blocks=args.n_conv_blocks,
        cnn_out_channels=args.cnn_out_channels,
        cnn_dropout=args.cnn_dropout,
        lstm_num_layers=args.lstm_num_layers,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_bidirectional=args.lstm_bidirectional,
        lstm_dropout=args.lstm_dropout
    ).to(device)

    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.num_epochs,
                    model_name=args.model_name)
    print('Model trained. Final validation accuracy:', history['val_acc'][-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with given parameters.")

    parser.add_argument("--n_conv_blocks", type=check_max, default=2,
                        help="Number of convolutional blocks in the model. Maximum of 7.")
    parser.add_argument("--cnn_out_channels", type=int, default=32,
                        help="Output size of the convolution layers in the model.")
    parser.add_argument("--cnn_dropout", type=float, default=0.388,
                        help="Dropout probability between the CNN blocks.")

    parser.add_argument("--lstm_bidirectional", action="store_true",
                        help="If set, the model will be bidirectional.")
    parser.add_argument("--lstm_num_layers", type=int, default=3,
                        help="Number of layers in the LSTM.")
    parser.add_argument("--lstm_hidden_size", type=int, default=174,
                        help="Size of a hidden layer in the LSTM.")
    parser.add_argument("--lstm_dropout", type=float, default=0.2689664397393872,
                        help="Dropout probability between the LSTM layers.")

    parser.add_argument("--learning_rate", type=float, default=0.009728535043244946,
                        help="Learning rate for training the model.")
    parser.add_argument("--weight_decay", type=float, default=7e-05,
                        help="Weight decay in the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=250,
                        help="Number of epochs to train the model.")
    parser.add_argument("--verbose", action="store_true",
                        help="If set, will print the information to the terminal.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name of the model where the weights will be saved.")
    args = parser.parse_args()

    print("Training model with parameters:")
    pprint(vars(args))
    print("-" * 30)

    main(args)

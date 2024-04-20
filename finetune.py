from os.path import join

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import random

from utils import load_data, stack_data, stack_windowed_data
from dataset_loaders import create_dataloaders, create_cnn_dataloaders
from models import LSTMModel, CNNModel, HybridModel

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed_all(seed)
else:
    device = 'cpu'



data = load_data(file_path=join('data', 'W_AUGMENTED_DATA.json'))
sequences, targets = stack_data(data)

def objective_lstm(trial):
    # batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    # lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 128, 300)
    # lstm_bidirectional = trial.suggest_categorical('lstm_bidirectional', [True, False])
    # lstm_bidirectional = True      # fine-tuning uni- and bidirectional LSTMs separately
    # lstm_dropout = trial.suggest_float('lstm_dropout', 0.2, 0.7)
    # lstm_num_layers = trial.suggest_int('lstm_num_layers', 2, 5)

    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    train_loader, val_loader, test_loader, class_weights = create_dataloaders(sequences, targets, batch_size=186,
                                                                              return_class_weights=True,
                                                                              verbose=False)

    model = LSTMModel(lstm_hidden_size=231,
                      lstm_dropout=0.5830300061685474,
                      lstm_num_layers=2,
                      lstm_bidirectional=False).to(device)

    class_weights_tensor = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = 75
    avg_val_acc = 0
    avg_val_loss = 10000
    for epoch in range(num_epochs):
        model.train()
        for padded_sequences, lengths, labels in train_loader:
            padded_sequences = padded_sequences.to(device)

            optimizer.zero_grad()
            outputs = model(padded_sequences.float().to(device), lengths)

            loss = criterion(outputs.view(-1, 4), labels.long().view(-1).to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        val_acc = []
        val_loss = []

        with torch.no_grad():
            for padded_sequences, lengths, labels in val_loader:
                outputs = model(padded_sequences.float().to(device), lengths)
                loss = criterion(outputs.view(-1, 4), labels.long().view(-1).to(device))
                val_loss.append(loss.item())
                _, predicted = torch.max(outputs.view(-1, 4), 1)
                accuracy = accuracy_score(labels.view(-1).numpy(), predicted.cpu().detach().numpy())
                val_acc.append(accuracy)

        avg_val_acc = np.mean(val_acc)
        avg_val_loss = np.mean(val_loss)
        trial.report(avg_val_loss, epoch)

        # prune experiment based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss



def objective_cnn(trial):
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    cnn_dropout = trial.suggest_float('cnn_dropout', 0.2, 0.7)
    n_blocks = trial.suggest_int('n_blocks', 1, 5)
    out_channels = trial.suggest_categorical('out_channels', [16, 32, 64, 128])

    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    train_loader, val_loader, test_loader, class_weights = create_cnn_dataloaders(signals, targets,
                                                                                  batch_size=batch_size,
                                                                                  return_class_weights=True,
                                                                                  verbose=False)

    model = CNNModel(
        cnn_dropout=cnn_dropout,
        n_blocks=n_blocks,
        out_channels=out_channels
    ).to(device)

    class_weights_tensor = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = 150
    avg_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences.float().to(device))

            loss = criterion(outputs.view(-1, 4), labels.long().view(-1).to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        val_acc = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = model(sequences.float().to(device))

                _, predicted = torch.max(outputs.view(-1, 4), 1)
                accuracy = accuracy_score(labels.view(-1).numpy(), predicted.cpu().detach().numpy())
                val_acc.append(accuracy)

        avg_val_acc = np.mean(val_acc)

        trial.report(avg_val_acc, epoch)

        # prune experiment based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_acc


# data = load_data(file_path=join('data', 'W_AUGMENTED_DATA.json'))
# signals, targets = stack_windowed_data(data)
def objective_hybrid(trial):
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    cnn_dropout = trial.suggest_float('cnn_dropout', 0.2, 0.7)
    n_blocks = trial.suggest_int('n_blocks', 1, 5)
    out_channels = trial.suggest_categorical('out_channels', [16, 32, 64, 128])

    lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 128, 300)
    lstm_bidirectional = trial.suggest_categorical('lstm_bidirectional', [True, False])
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.2, 0.7)
    lstm_num_layers = trial.suggest_int('lstm_num_layers', 2, 5)

    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    train_loader, val_loader, test_loader, class_weights = create_dataloaders(signals, targets,
                                                                              batch_size=batch_size,
                                                                              return_class_weights=True,
                                                                              verbose=False)

    model = HybridModel(
        cnn_dropout=cnn_dropout,
        n_conv_blocks=n_blocks,
        cnn_out_channels=out_channels,
        lstm_hidden_size=lstm_hidden_size,
        lstm_dropout=lstm_dropout,
        lstm_num_layers=lstm_num_layers,
        lstm_bidirectional=lstm_bidirectional
    ).to(device)

    class_weights_tensor = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = 150
    avg_val_loss = 0

    for epoch in range(num_epochs):
        model.train()
        for padded_sequences, lengths, labels in train_loader:
            padded_sequences = padded_sequences.to(device)

            optimizer.zero_grad()
            outputs = model(padded_sequences.float().to(device), lengths)

            loss = criterion(outputs.view(-1, 4), labels.long().view(-1).to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = []

        with torch.no_grad():
            for padded_sequences, lengths, labels in val_loader:
                outputs = model(padded_sequences.float().to(device), lengths)
                loss = criterion(outputs.view(-1, 4), labels.long().view(-1).to(device))
                val_loss.append(loss.item())
                # _, predicted = torch.max(outputs.view(-1, 4), 1)
                # accuracy = accuracy_score(labels.view(-1).numpy(), predicted.cpu().detach().numpy())
                # val_acc.append(accuracy)

        avg_val_loss = np.mean(val_loss)

        trial.report(avg_val_loss, epoch)

        # prune experiment based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')

    # model = 'cnn'
    #
    # if model == 'lstm':
    #     data = load_data(file_path=join('data', 'W_AUGMENTED_DATA.json'))
    #     sequences, targets = stack_data(data)
    #     study.optimize(objective_lstm, n_trials=100)
    # else:
    # data = load_data(file_path=join('data', 'W_AUGMENTED_DATA.json'))
    # sequences, targets = stack_windowed_data(data)
    study.optimize(objective_lstm, n_trials=25)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Chosen parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

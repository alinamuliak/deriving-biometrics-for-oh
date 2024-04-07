import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import random

from utils import load_data, stack_data
from dataset_loaders import create_dataloaders
from models import LSTMModel

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed_all(seed)
else:
    device = 'cpu'


data = load_data()
sequences, targets = stack_data(data)


def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 128, 300)
    # lstm_bidirectional = trial.suggest_categorical('lstm_bidirectional', [True, False])
    lstm_bidirectional = True      # fine-tuning uni- and bidirectional LSTMs separately
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.2, 0.7)
    lstm_num_layers = trial.suggest_int('lstm_num_layers', 2, 5)

    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    train_loader, val_loader, test_loader, class_weights = create_dataloaders(sequences, targets, batch_size=batch_size,
                                                                              return_class_weights=True,
                                                                              verbose=False)

    model = LSTMModel(
        lstm_hidden_size=lstm_hidden_size,
        lstm_bidirectional=lstm_bidirectional,
        lstm_dropout=lstm_dropout,
        lstm_num_layers=lstm_num_layers
    ).to(device)

    class_weights_tensor = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = 50
    avg_val_acc = 0
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

        with torch.no_grad():
            for padded_sequences, lengths, labels in val_loader:
                outputs = model(padded_sequences.float().to(device), lengths)
                _, predicted = torch.max(outputs.view(-1, 4), 1)
                accuracy = accuracy_score(labels.view(-1).numpy(), predicted.cpu().detach().numpy())
                val_acc.append(accuracy)

        avg_val_acc = np.mean(val_acc)
        trial.report(avg_val_acc, epoch)

        # prune experiment based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_acc


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Chosen parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

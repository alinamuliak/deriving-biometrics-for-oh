import json
import os
import argparse
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import random
import matplotlib.pyplot as plt
from os.path import join

from typing import Callable
from torch.utils.data import DataLoader

from train import train
from utils import load_data, stack_data, stack_windowed_data
from dataset_loaders import create_dataloaders
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


def main(args):
    data = load_data(file_path=join('data', 'W_AUGMENTED_DATA.json'))
    sequences, targets = stack_data(data)
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(sequences, targets,
                                                                              batch_size=args.batch_size,
                                                                              return_class_weights=True,
                                                                              verbose=False)
    class_weights_tensor = class_weights.to(device)

    model = LSTMModel(lstm_hidden_size=args.hidden_size,
                      lstm_dropout=args.dropout,
                      lstm_num_layers=args.num_layers,
                      lstm_bidirectional=args.bidirectional).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.num_epochs,
                    model_name=args.model_name)
    print('Model trained. Final validation accuracy:', history['val_acc'][-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with given parameters.")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training the model.")
    parser.add_argument("--bidirectional", action="store_true",
                        help="If set, the model will be bidirectional.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers in the LSTM.")
    parser.add_argument("--hidden_size", type=int, default=231,
                        help="Size of a hidden layer in the LSTM.")
    parser.add_argument("--dropout", type=float, default=0.583,
                        help="Dropout probability between the LSTM layers.")

    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Learning rate for training the model.")
    parser.add_argument("--weight_decay", type=float, default=9e-06,
                        help="Weight decay in the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=200,
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

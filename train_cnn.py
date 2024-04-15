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
from dataset_loaders import create_dataloaders, create_cnn_dataloaders
from models import CNNModel, LSTMModel
from train import train

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
    sequences, targets = stack_windowed_data(data)
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(sequences, targets,
                                                                              batch_size=args.batch_size,
                                                                              return_class_weights=True,
                                                                              verbose=False)
    class_weights_tensor = class_weights.to(device)

    model = CNNModel(
        cnn_dropout=args.dropout,
        n_blocks=args.n_conv_blocks,
        out_channels=args.out_channels
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.num_epochs,
                    model_name=args.model_name)
    print('Model trained. Final validation accuracy:', history['val_acc'][-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with given parameters.")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training the model.")

    parser.add_argument("--n_conv_blocks", type=check_max, default=2,
                        help="Number of convolutional blocks in the model. Maximum of 7.")
    parser.add_argument("--out_channels", type=int, default=32,
                        help="Output size of the convolution layers in the model.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout probability between the CNN blocks.")

    parser.add_argument("--learning_rate", type=float, default=5e-05,
                        help="Learning rate for training the model.")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="Weight decay in the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=300,
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

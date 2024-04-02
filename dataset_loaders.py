"""This module contains classes and functions for creating torch.Dataset and torch.DataLoader instances."""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    """
    Dataset class for signals data.

    Parameters:
        sequences (list): A list containing sequences of signals.
        targets (list): A list containing labels corresponding to each sequence.
        test_dataset (bool, optional): Whether the dataset is for testing. Defaults to False.
        windowed_data (bool, optional): Whether the data is windowed. Defaults to False.

    Attributes:
        X (list): A list containing sequences of signals.
        y (list): A list containing labels corresponding to each sequence.
        test_dataset (bool): Whether the dataset is for testing.
        windowed_data (bool): Whether the data is windowed.
    """
    def __init__(self, sequences: list, targets: list, test_dataset: bool = False, windowed_data: bool = False):
        """
        Initialize SequenceDataset with sequences (signals data), targets, and optional flags.

        :param sequences: A list containing sequences of signals.
        :param targets: A list containing labels corresponding to each sequence.
        :param test_dataset: Whether the dataset is for testing. Defaults to False.
        :param windowed_data: Whether the data is windowed. Defaults to False.
        """
        self.X = sequences
        self.y = targets
        self.test_dataset = test_dataset
        self.windowed_data = windowed_data

    def __len__(self):
        """
        Get the length of the dataset.

        :return: The number of sequences in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get an item from the dataset.

        :param idx: Index of the item to retrieve.
        :return: A tuple containing the sequence and its label.
        """
        if self.test_dataset:
            return self.X[idx], self.y[idx]

        if self.windowed_data:
            return self.X[idx][:, :3, :], self.y[idx]

        return self.X[idx][:, :3], self.y[idx]


def collate_fn(batch) -> tuple:
    """
    Collate function for padding the data.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])

    padded_sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True)
    padded_labels = pad_sequence([torch.tensor(l) for l in labels], batch_first=True)

    lengths, perm_idx = lengths.sort(0, descending=True)
    padded_sequences = padded_sequences[perm_idx]
    padded_labels = padded_labels[perm_idx]

    return padded_sequences, lengths, padded_labels


def create_dataloaders(X: list, y: list, batch_size: int, seed: int = 7777777,
                       return_class_weights: bool = True, verbose: bool = False)\
        -> tuple[DataLoader, DataLoader, DataLoader] | tuple[DataLoader, DataLoader, DataLoader, torch.FloatTensor]:
    """
    Splits the data for train, validation, and test datasets.
    Creates dataloaders for each with the specified batch_size.

    :param X: List of sequences of signals.
    :param y: List of labels.
    :param batch_size: Batch size for dataloaders.
    :param seed: Random seed for data splitting and shuffling.
    :param return_class_weights: Whether to return an array with class weights from train dataset. Defaults to True.
    :param verbose: Whether to print progress during dataloaders creation.

    :return: Train, validation, and test dataloaders. If return_class_weights is True, returns this array, too.
    """
    generator = torch.Generator().manual_seed(seed)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.175, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.175, random_state=seed)

    if verbose:
        print("Created train, validation, and test datasets.")
        print(f"X_train length: {len(X_train)}, X_val length: {len(X_val)}, and X_test length: {len(X_test)}.")

    train_dataset = SequenceDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, generator=generator)

    val_dataset = SequenceDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, generator=generator)

    test_dataset = SequenceDataset(X_test, y_test, test_dataset=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, generator=generator)

    if verbose:
        print(f"Created dataloaders with batch size {batch_size}")

    if return_class_weights:
        class_counts = np.bincount(np.concatenate(y_train).ravel())
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        class_weights_tensor = torch.FloatTensor(class_weights)

        return train_loader, val_loader, test_loader, class_weights_tensor

    return train_loader, val_loader, test_loader

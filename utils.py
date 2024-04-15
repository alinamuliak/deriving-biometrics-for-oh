"""This module contains helper functions for working with the data."""
import argparse
import json
import numpy as np


def load_data(file_path: str = "data/DATA.json") -> dict:
    """
    Load data from specified file as a dictionary.
    :param file_path: Path to the .json file.
    :return: Dictionary of signals containing ACC, PPG and HR data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def stack_data(data: dict) -> tuple:
    """
    Stack signals from loaded data.
    :param data: Loaded data containing signals.
    :return: Tuple containing stacked data and corresponding targets.
    """
    sequences = []

    for i in range(len(data['ppg_data'])):
        stacked_data = np.stack((data['ppg_data'][i],
                                 data['hr_data'][i],
                                 data['accelerometer_data'][i],
                                 data['ppg_not_normalized'][i],
                                 data['hr_not_normalized'][i]),
                                axis=1)
        sequences.append(stacked_data)

    return sequences, data['targets']


def create_windows(signal_data, label_data, window_size, step_size) -> tuple[np.ndarray, np.ndarray]:
    """
    Split signals into windows of size window_size with stride step_size.
    :param signal_data: Array with stacked signals.
    :param label_data: Array containing labels.
    :param window_size: Size of the window.
    :param step_size: Stride for splitting the signals.
    :return: Tuple of numpy arrays containing data split into windows and corresponding labels.
    """
    windows = []
    window_labels = []
    for i in range(len(signal_data)):
        signal, labels = signal_data[i], label_data[i]
        for start in range(0, len(signal) - window_size + 1, step_size):
            end = start + window_size
            window = signal[start:end]
            label = np.bincount(labels[start: end]).argmax()
            windows.append(window)
            window_labels.append(label)

    return np.array(windows), np.array(window_labels)


def stack_windowed_data(data: dict, window_size: int = 150, step_size: int = 75) -> tuple:
    """
    Create windows and stack the data.
    :param data:
    :param window_size:
    :param step_size:
    :return:
    """
    sequences, sequence_labels = [], []

    for i in range(len(data['ppg_data'])):
        ppg, labels = create_windows([data['ppg_data'][i]], [data['targets'][i]],
                                     window_size, step_size)
        ppg_nn, _ = create_windows([data['ppg_not_normalized'][i]], [data['targets'][i]],
                                   window_size, step_size)
        hr_nn, _ = create_windows([data['hr_not_normalized'][i]], [data['targets'][i]],
                                  window_size, step_size)
        hr, _ = create_windows([data['hr_data'][i]], [data['targets'][i]],
                               window_size, step_size)
        acc, _ = create_windows([data['accelerometer_data'][i]], [data['targets'][i]],
                                window_size, step_size)

        combined_windows = np.stack((ppg, hr, acc, ppg_nn, hr_nn), axis=1)
        sequences.append(combined_windows)
        sequence_labels.append(labels)
    return sequences, sequence_labels

def check_max(x):
    """
    Checks if the provided value is in the range [0, 7]. Is used for argument parsing.
    :param x:
    :return:
    """
    x = int(x)
    if x > 7:
        raise argparse.ArgumentTypeError('Maximum allowed value for n_conv_blocks is 7.')
    return x
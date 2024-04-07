# todo: add documentation

import json
from copy import deepcopy

import numpy as np
import random

from tqdm import tqdm
from numpy.typing import ArrayLike
from sklearn.preprocessing import MinMaxScaler


# Adapted https://maddevs.io/writeups/basic-data-augmentation-method-applied-to-time-series/
def add_gaussian_noise(time_series: list[np.ndarray], mean: float = 0.0, std_factor: float = 0.1) -> list:
    """
    Adds Gaussian noise to a time series.

    :param time_series: A time series to which noise is added, array-like.
    :param mean: The average value of the noise, float. Default is 0.0.
    :param std_factor: Standard deviation of noise, float.
                       The std of the signal is multiplied with it to get the std of the noise. Default is 0.1.

    :return noisy_series: List of time series with added noise.
    """
    noisy_series = []
    for i in range(len(time_series)):
        std = np.std(time_series[i])
        noise = np.random.normal(mean, std * std_factor, len(time_series[i]))
        noisy_series.append((time_series[i] + noise).tolist())

    return noisy_series


# Adapted from https://maddevs.io/writeups/basic-data-augmentation-method-applied-to-time-series/
def time_warping(time_series: list[np.ndarray], labels: np.ndarray, num_operations: int = 50)\
        -> tuple[list, list]:
    """
    Applying time warping to a time series.

    :param time_series: Time series, numpy array.
    :param labels: Corresponding labels, numpy array.
    :param num_operations: Number of insert/delete operations, int.
    :return: Distorted time series.
    """
    warped_series = time_series.copy()
    warped_labels = labels.copy()
    for _ in range(num_operations):
        operation_type = random.choice(["insert", "delete"])
        index = random.randint(1, len(warped_series) - 2)

        if operation_type == "insert":
            insert_points_num = random.randint(1, 20)
            for feature_i in range(len(warped_series)):
                for j in range(insert_points_num):
                    # Insert a value by interpolating between two adjacent points
                    insertion_value = (warped_series[feature_i][index + j - 1] + warped_series[feature_i][index + j]) * 0.5
                    warped_series[feature_i] = np.insert(warped_series[feature_i], index, insertion_value)

            for j in range(insert_points_num):
                # Insert a previous label
                insertion_value = warped_labels[index + j - 1]
                warped_labels = np.insert(warped_labels, index, insertion_value)

        elif operation_type == "delete":
            delete_points_num = random.randint(1, 20)
            # Remove random points
            for feature_i in range(len(warped_series)):
                for j in range(delete_points_num):
                    warped_series[feature_i] = np.delete(warped_series[feature_i], index + j)

            for j in range(delete_points_num):
                # Delete label
                warped_labels = np.delete(warped_labels, index + j)

        else:
            raise ValueError("Invalid operation type")

    return np.array(warped_series).tolist(), warped_labels.tolist()


def apply_random_augmentation(time_series: list[np.ndarray], labels: np.ndarray) -> tuple[list, list]:
    transform_type = random.choice(["noise", "time-warping", "both"])
    if transform_type == "noise":
        transformed_signals = add_gaussian_noise(time_series)
        return transformed_signals, labels.tolist()

    if transform_type == "time-warping":
        # for time-warping the labels should change too
        transformed_signals, transformed_labels = time_warping(time_series, labels)
        return transformed_signals, transformed_labels

    if transform_type == "both":
        transformed_signals, transformed_labels = time_warping(time_series, labels)
        transformed_signals = add_gaussian_noise(transformed_signals)
        return transformed_signals, transformed_labels


def normalize_signal(sig: ArrayLike) -> list:
    scaler = MinMaxScaler(feature_range=(0, 1))
    sig_norm = scaler.fit_transform([[x] for x in sig]).reshape(len(sig))
    return sig_norm.tolist()


def augment_data(data: dict[str, list], augment_percentage: float = 0.3, save_to_file: str | None = None) -> dict[str, list]:
    augmented_data = deepcopy(data)

    dataset_size = len(data['ppg_data'])
    signals_to_augment = random.sample(list(np.arange(dataset_size)), int(dataset_size * augment_percentage))
    for i in tqdm(signals_to_augment):
        signals = [np.array(data['ppg_not_normalized'][i]),
                   np.array(data['hr_not_normalized'][i]),
                   np.array(data['accelerometer_data'][i])]
        augmented_signals, augmented_label = apply_random_augmentation(signals, np.array(data['targets'][i]))

        augmented_data['ppg_not_normalized'].append(augmented_signals[0])
        augmented_data['hr_not_normalized'].append(augmented_signals[1])

        normalized_ppg = normalize_signal(augmented_signals[0])
        normalized_hr = normalize_signal(augmented_signals[1])
        normalized_acc = normalize_signal(augmented_signals[2])
        augmented_data['ppg_data'].append(normalized_ppg)
        augmented_data['hr_data'].append(normalized_hr)
        augmented_data['accelerometer_data'].append(normalized_acc)
        augmented_data['targets'].append(augmented_label)

    if save_to_file is not None:
        with open(save_to_file, 'w') as f:
            json.dump(augmented_data, f, indent=4)

    return augmented_data


if __name__ == '__main__':
    from utils import load_data
    from os.path import join
    data = load_data()
    path_to_save = join('data', 'W_AUGMENTED_DATA.json')
    augmented_data = augment_data(data, save_to_file=path_to_save)
    print(f"Successfully created dataset with augmentation! "
          f"Initial dataset size: {len(data['ppg_data'])}, augmented dataset size: {len(augmented_data['ppg_data'])}")
    if path_to_save:
        print(f'Augmented dataset saved to {path_to_save}.')

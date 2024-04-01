"""
This module contains functions for biometrics calculations and comparison between the true and estimated biometrics.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error


def calculate_ohv1_mae(signals_for_biometrics: dict) -> float:
    """
    Calculates the mean absolute error (MAE) between true and estimated Orthostatic Hypovolemia 1 (OHV1) biometric.
    :param signals_for_biometrics: A dictionary containing signals, true and estimated stages labels.
    :return: MAE of the OHV1.
    """
    ohv1_true_array, ohv1_pred_array = [], []
    for signal in signals_for_biometrics:
        try:
            mask = np.where(np.array(signal['true_labels']) == 0)
            min_supine = min(signal['ppg'][mask])
            mask = np.where(np.array(signal['true_labels']) == 2)
            min_standing = min(signal['ppg'][mask])
            ohv1_true = abs(min_standing - min_supine)

            mask = np.where(np.array(signal['predicted_labels']) == 0)
            min_supine = min(signal['ppg'][mask])
            mask = np.where(np.array(signal['predicted_labels']) == 2)
            min_standing = min(signal['ppg'][mask])

            ohv1_true_array.append(ohv1_true)
            ohv1_pred_array.append(abs(min_standing - min_supine))
        except ValueError:
            continue

    return mean_absolute_error(ohv1_true_array, ohv1_pred_array)


def calculate_ohv2_mae(signas_for_biometrics: dict) -> float:
    """
    Calculates the mean absolute error (MAE) between true and estimated Orthostatic Hypovolemia 2 (OHV2) biometric.
    :param signals_for_biometrics: A dictionary containing signals, true and estimated stages labels.
    :return: MAE of the OHV2.
    """
    ohv2_true_array, ohv2_pred_array = [], []

    for signal in signas_for_biometrics:
        try:
            mask = np.where(np.array(signal['true_labels']) == 0)
            min_supine = min(signal['ppg'][mask])
            mask = np.where(np.array(signal['true_labels']) == 3)
            min_standing = min(signal['ppg'][mask])
            ohv2_true = abs(min_standing - min_supine)

            mask = np.where(np.array(signal['predicted_labels']) == 0)
            min_supine = min(signal['ppg'][mask])
            mask = np.where(np.array(signal['predicted_labels']) == 3)
            min_standing = min(signal['ppg'][mask])

            ohv2_true_array.append(ohv2_true)
            ohv2_pred_array.append(abs(min_standing - min_supine))
        except ValueError:
            continue

    return mean_absolute_error(ohv2_true_array, ohv2_pred_array)


def calculate_otc_mae(signals_for_biometrics: dict, sampling_rate: int = 50) -> float:
    """
    Calculates the mean absolute error (MAE) between true and estimated
    Orthostatic Time Constraint (OTC) biometric.

    :param signals_for_biometrics: A dictionary containing signals, true and estimated stages labels.
    :param sampling_rate: The sampling rate of the signals. Defaults to 50.
    :return: MAE of the OTC.
    """
    otc_true_array, otc_pred_array = [], []

    for signal in signals_for_biometrics:
        try:
            start_transition = signal['true_labels'].index(1)
            orthostatis_archieved_at = signal['true_labels'].index(2)
            otc_true = abs(orthostatis_archieved_at - start_transition) / sampling_rate

            start_transition = signal['predicted_labels'].index(1)
            orthostatis_archieved_at = signal['predicted_labels'].index(2)

            otc_true_array.append(otc_true)
            otc_pred_array.append(abs(orthostatis_archieved_at - start_transition) / sampling_rate)
        except ValueError:
            continue

    return mean_absolute_error(otc_true_array, otc_pred_array)


def calculate_pot_mae(signals_for_biometrics):
    """
    Calculates the mean absolute error (MAE) between true and estimated
    Postural Orthostatic Tachycardia (POT) biometric.

    :param signals_for_biometrics: A dictionary containing signals, true and estimated stages labels.
    :return: MAE of the POT.
    """

    pot_true_array, pot_pred_array = [], []
    for signal in signals_for_biometrics:
        try:
            mask = np.where(np.array(signal['true_labels']) == 0)
            avg_supine = np.mean(signal['hr'][mask])
            mask = np.where((np.array(signal['true_labels']) == 2) | (np.array(signal['true_labels']) == 3))
            max_standing_orthostatis = max(signal['hr'][mask])
            pot_true = abs(max_standing_orthostatis - avg_supine)

            mask = np.where(np.array(signal['predicted_labels']) == 0)
            avg_supine = np.mean(signal['hr'][mask])
            mask = np.where((np.array(signal['predicted_labels']) == 2) | (np.array(signal['predicted_labels']) == 3))
            max_standing_orthostatis = max(signal['hr'][mask])

            pot_true_array.append(pot_true)
            pot_pred_array.append(abs(max_standing_orthostatis - avg_supine))

        except:
            continue

    return mean_absolute_error(pot_true_array, pot_pred_array)

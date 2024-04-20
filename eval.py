import argparse
import sys
from pprint import pprint

import torch
from tabulate import tabulate
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from biometrics import calculate_mean_absolute_error_all, calculate_percentage_error_all
from dataset_loaders import create_dataloaders
from utils import load_data, stack_windowed_data, stack_data, check_model_type


def calculate_biometrics_error_for_windowed_data(test_loader, model, device: str = 'cpu',
                                                 window_size: int = 150, step_size: float = 0.5):
    SIGNALS_FOR_BIOMETRICS_all = []
    NUMBER_OF_WINDOWS_PER_SIGNAL = 58

    for padded_sequences, lengths, labels in test_loader:

        for record_num in range(len(padded_sequences)):
            outputs = model(padded_sequences[:, :, :3, :].float().to(device), lengths)
            _, predicted = torch.max(outputs.view(-1, 4), 1)

            current_len = lengths[record_num]
            windows, true_labels, window_predictions = (padded_sequences[record_num][:current_len],
                                                        labels[record_num][:current_len],
                                                        predicted[record_num * NUMBER_OF_WINDOWS_PER_SIGNAL:
                                                                  record_num * NUMBER_OF_WINDOWS_PER_SIGNAL
                                                                  + NUMBER_OF_WINDOWS_PER_SIGNAL][:current_len])

            signals_for_biometrics = {'ppg': np.array([]),
                                      'hr': np.array([]),
                                      'ppg_normalized': np.array([]),
                                      'true_labels': [],
                                      'predicted_labels': []}

            for i, (window, prediction) in enumerate(zip(windows, true_labels)):

                signals_for_biometrics['ppg'] = np.append(signals_for_biometrics['ppg'],
                                                          window.cpu()[3, :int(window_size * step_size)])
                signals_for_biometrics['hr'] = np.append(signals_for_biometrics['hr'],
                                                         window.cpu()[4, :int(window_size * step_size)])
                signals_for_biometrics['ppg_normalized'] = np.append(signals_for_biometrics['ppg_normalized'],
                                                                     window.cpu()[0, :int(window_size * step_size)])
                for _ in range(window_size - int(window_size * 0.5)):
                    signals_for_biometrics['true_labels'].append(prediction.item())

                if i == len(windows) - 1:
                    signals_for_biometrics['ppg'] = np.append(signals_for_biometrics['ppg'],
                                                              window.cpu()[3, int(window_size * step_size):])
                    signals_for_biometrics['hr'] = np.append(signals_for_biometrics['hr'],
                                                             window.cpu()[4, int(window_size * step_size):])
                    signals_for_biometrics['ppg_normalized'] = np.append(signals_for_biometrics['ppg_normalized'],
                                                                         window.cpu()[0, :int(window_size * step_size)])
                    for _ in range(window_size - int(window_size * 0.5)):
                        signals_for_biometrics['true_labels'].append(prediction.item())

            for i, (window, prediction) in enumerate(zip(windows, window_predictions)):

                for _ in range(window_size - int(window_size * 0.5)):
                    signals_for_biometrics['predicted_labels'].append(prediction.item())
                if i == len(windows) - 1:
                    for _ in range(window_size - int(window_size * 0.5)):
                        signals_for_biometrics['predicted_labels'].append(prediction.item())

            SIGNALS_FOR_BIOMETRICS_all.append(signals_for_biometrics)
    return calculate_mean_absolute_error_all(SIGNALS_FOR_BIOMETRICS_all), calculate_percentage_error_all(SIGNALS_FOR_BIOMETRICS_all)


def calculate_biometrics_error(test_loader, model, device: str = 'cpu'):
    SIGNALS_FOR_BIOMETRICS_all = []
    for padded_sequences, lengths, labels in test_loader:

        for record_num in range(len(padded_sequences)):
            outputs = model(padded_sequences[:, :, :3].float().to(device), lengths)
            _, predicted = torch.max(outputs.view(-1, 4), 1)

            current_len = lengths[record_num]
            sequence, true_labels, prediction = (padded_sequences[record_num][:current_len],
                                                 labels[record_num][:current_len],
                                                 predicted[:current_len].cpu())

            signals_for_biometrics = {'ppg': np.array([]),
                                      'hr': np.array([]),
                                      'ppg_normalized': np.array([]),
                                      'true_labels': [],
                                      'predicted_labels': []}

            signals_for_biometrics['ppg'] = np.append(signals_for_biometrics['ppg'], sequence.cpu()[:, 3])
            signals_for_biometrics['hr'] = np.append(signals_for_biometrics['hr'], sequence.cpu()[:, 4])
            signals_for_biometrics['ppg_normalized'] = np.append(signals_for_biometrics['ppg_normalized'],
                                                                 sequence.cpu()[:, 0])

            for i in range(true_labels.size()[0]):
                signals_for_biometrics['true_labels'].append(true_labels.numpy()[i])

            for i in range(prediction.size()[0]):
                signals_for_biometrics['predicted_labels'].append(prediction.numpy()[i])

            SIGNALS_FOR_BIOMETRICS_all.append(signals_for_biometrics)

    return calculate_mean_absolute_error_all(SIGNALS_FOR_BIOMETRICS_all), calculate_percentage_error_all(SIGNALS_FOR_BIOMETRICS_all)


def print_results_as_table(accuracy: float, f1: float, biometrics_mae: list[float], biometrics_mpe: list[float]) -> None:
    if np.any(np.isnan(biometrics_mae)) or np.any(np.isnan(biometrics_mpe)):
        print('\nNOTE: Some of the evaluation metrics were nan. '
              'It is allowed behavior which typically means that '
              'the model does not classify the phases needed for the metrics calculation at all.')

    print('Finished! Results:')
    print(tabulate([
        ['Accuracy', f'{round(accuracy * 100, 2)}%'],
        ['F1 score', f'{round(f1 * 100, 2)}%'],
        ['OHV1 MAE', f'{round(biometrics_mae[0], 2)} a.u.'],
        ['OHV2 MAE', f'{round(biometrics_mae[1], 2)} a.u.'],
        ['OTC MAE',  f'{round(biometrics_mae[2], 2)} sec'],
        ['POT MAE',  f'{round(biometrics_mae[3], 2)} bpm'],

        ['OHV1 MPE', f'{round(biometrics_mpe[0], 2)} %'],
        ['OHV2 MPE', f'{round(biometrics_mpe[1], 2)} %'],
        ['OTC MPE',  f'{round(biometrics_mpe[2], 2)} %'],
        ['POT MPE',  f'{round(biometrics_mpe[3], 2)} %']
    ], headers=['Metric', 'Value'], tablefmt='orgtbl'))


def eval_models_windowed_data(model, batch_size: int, device: str = 'cpu', save_plots_to: str | None = None):
    data = load_data(file_path='data/W_AUGMENTED_DATA.json')
    sequences, labels = stack_windowed_data(data)
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(sequences, labels, batch_size=batch_size,
                                                                              return_class_weights=True,
                                                                              verbose=False)


    alll = {'true': np.array([]), 'pred': np.array([])}
    with torch.no_grad():
        for padded_sequences, lengths, labels in test_loader:
            padded_sequences = padded_sequences[:, :, :3]
            outputs = model(padded_sequences.float().to(device), lengths)
            _, predicted = torch.max(outputs.view(-1, 4), 1)
            alll['true'] = np.append(alll['true'], labels)
            alll['pred'] = np.append(alll['pred'], predicted.cpu().detach().numpy())

    accuracy = accuracy_score(alll['true'], alll['pred'])
    f1 = f1_score(alll['true'], alll['pred'], average='weighted')
    # print(f'Accuracy: {round(accuracy * 100, 2)}%, F1 score: {round(f1 * 100, 2)}%')

    if save_plots_to is not None:
        cm = confusion_matrix(alll['true'], alll['pred'], normalize='pred')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['Supine', 'Transition', 'Standing', 'Orthostatic'])
        fig, ax = plt.subplots()
        ax.grid(False)
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        fig.savefig(f'{save_plots_to}/confusion-matrix.png', bbox_inches='tight')

    biometrics_mae, biometrics_mpe = calculate_biometrics_error_for_windowed_data(test_loader, model, device)
    print_results_as_table(accuracy, f1, biometrics_mae, biometrics_mpe)


def eval_lstm(model, batch_size: int, device: str = 'cpu', save_plots_to: str | None = None):
    data = load_data(file_path='data/W_AUGMENTED_DATA.json')
    sequences, labels = stack_data(data)
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(sequences, labels, batch_size=batch_size,
                                                                              return_class_weights=True,
                                                                              verbose=False)


    alll = {'true': np.array([]), 'pred': np.array([])}
    with torch.no_grad():
        for padded_sequences, lengths, labels in test_loader:
            padded_sequences = padded_sequences[:, :, :3]
            outputs = model(padded_sequences.float().to(device), lengths)
            _, predicted = torch.max(outputs.view(-1, 4), 1)
            alll['true'] = np.append(alll['true'], labels)
            alll['pred'] = np.append(alll['pred'], predicted.cpu().detach().numpy())

    accuracy = accuracy_score(alll['true'], alll['pred'])
    f1 = f1_score(alll['true'], alll['pred'], average='weighted')

    if save_plots_to is not None:
        cm = confusion_matrix(alll['true'], alll['pred'], normalize='pred')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['Supine', 'Transition', 'Standing', 'Orthostatic'])
        fig, ax = plt.subplots()
        ax.grid(False)
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        fig.savefig(f'{save_plots_to}/confusion-matrix.png', bbox_inches='tight')

    biometrics_mae, biometrics_mpe = calculate_biometrics_error_for_windowed_data(test_loader, model, device)
    print_results_as_table(accuracy, f1, biometrics_mae, biometrics_mpe)


def main(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print('CUDA is not available. Falling back to CPU.')

    if args.model_type in ['cnn', 'hybrid']:
        model = torch.load(args.chkpt_path, map_location=args.device)
        check_model_type(model, args.model_type)

        eval_models_windowed_data(model, args.batch_size, args.device, args.save_plots_to)

    elif args.model_type == 'lstm':
        model = torch.load(args.chkpt_path, map_location=args.device)
        check_model_type(model, args.model_type)

        eval_lstm(model, args.batch_size, args.device, args.save_plots_to)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with given parameters.")
    parser.add_argument('--model_type',
                        choices=['cnn', 'lstm', 'hybrid'],
                        required=True,
                        help='Model choice to be evaluated.')
    parser.add_argument('--chkpt_path',
                        required=True,
                        help='Path where the trained model was saved.')
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Batch size used for training the model.")

    parser.add_argument("--device", type=str, required=False, default='cuda',
                        help="Device to be used for the dataloader and model.")
    parser.add_argument("--save_plots_to", type=str, required=False, default=None,
                        help="Path to save plots. If not specified, the plots are not saved.")

    args = parser.parse_args()

    print("Evaluating...", end=" ")

    main(args)

"""This module contains model classes."""
# TODO: add documentation

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CNNModel(nn.Module):
    def __init__(self, num_classes=4,
                 cnn_dropout=0.5,
                 **kwargs):
        super(CNNModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.dropout = nn.Dropout(p=cnn_dropout)
        cnn_output_size = 2368

        self.fc = nn.Linear(cnn_output_size, num_classes)

    def forward(self, x):
        batch_size, num_windows, channels, window_size = x.shape
        x = x.view(-1, channels, window_size)
        cnn_out = self.cnn(x)
        cnn_out = self.dropout(cnn_out)
        cnn_out = cnn_out.view(batch_size, num_windows, -1)

        final_out = self.fc(cnn_out)
        return final_out


class HybridModel(nn.Module):
    def __init__(self, num_classes=4,
                 cnn_dropout=0.5,
                 lstm_hidden_size=128,
                 lstm_bidirectional=True,
                 lstm_dropout=0.5,
                 lstm_num_layers=2,
                 **kwargs):
        super(HybridModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.dropout = nn.Dropout(p=cnn_dropout)
        cnn_output_size = 2368

        self.lstm = nn.LSTM(input_size=cnn_output_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bidirectional=lstm_bidirectional,
                            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
                            batch_first=True)

        self.fc = nn.Linear(lstm_hidden_size * 2 if lstm_bidirectional else lstm_hidden_size, num_classes)

    def forward(self, x, lengths):
        batch_size, num_windows, channels, window_size = x.shape
        x = x.view(-1, channels, window_size)
        cnn_out = self.cnn(x)
        cnn_out = self.dropout(cnn_out)
        cnn_out = cnn_out.view(batch_size, num_windows, -1)

        x_packed = pack_padded_sequence(cnn_out, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x_packed)
        output, output_lengths = pad_packed_sequence(lstm_out, batch_first=True)

        final_out = self.fc(output)
        return final_out


class LSTMModel(nn.Module):
    def __init__(self, input_size=3, num_classes=4, lstm_hidden_size=128,
                 lstm_bidirectional=True, lstm_dropout=0.5, lstm_num_layers=2, **kwargs):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bidirectional=lstm_bidirectional,
                            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
                            batch_first=True)

        self.fc = nn.Linear(lstm_hidden_size * 2 if lstm_bidirectional else lstm_hidden_size, num_classes)

    def forward(self, x, lengths):
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x_packed)
        output, output_lengths = pad_packed_sequence(lstm_out, batch_first=True)
        final_out = self.fc(output)
        return final_out

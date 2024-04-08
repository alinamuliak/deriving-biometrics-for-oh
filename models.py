"""This module contains model classes."""
# TODO: add documentation

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CNNModel(nn.Module):
    def __init__(self, in_channels=3,
                 num_classes=4,
                 cnn_dropout=0.5,
                 n_blocks=2,
                 out_channels=32,
                 window_size=150
                 ):
        super(CNNModel, self).__init__()

        self.blocks = nn.ModuleList()

        for _ in range(n_blocks):
            block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.blocks.append(block)
            in_channels = out_channels

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=cnn_dropout)

        cnn_output_size = out_channels * (window_size // 2 ** n_blocks)
        self.fc = nn.Linear(cnn_output_size, num_classes)

    def forward(self, x):
        batch_size, window_size, channels = x.shape
        x = x.view(-1, channels, window_size)

        for block in self.blocks:
            x = block(x)

        x = self.flatten(x)
        x = self.dropout(x)

        final_out = self.fc(x)
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
            nn.Conv1d(3, 64, kernel_size=5, dilation=5, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.dropout = nn.Dropout(p=cnn_dropout)
        cnn_output_size = 4480

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
        # print(cnn_out.shape)

        x_packed = pack_padded_sequence(cnn_out, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x_packed)
        output, output_lengths = pad_packed_sequence(lstm_out, batch_first=True)

        final_out = self.fc(output)
        return final_out


class LSTMModel(nn.Module):
    def __init__(self, input_size=3, num_classes=4, lstm_hidden_size=128,
                 lstm_bidirectional=True, lstm_dropout=0.5, lstm_num_layers=2):
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

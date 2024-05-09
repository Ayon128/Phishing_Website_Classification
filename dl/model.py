import torch
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, cnn_filters, cnn_kernel_sizes):
        super(Model, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, num_filters, kernel_size)
            for num_filters, kernel_size in zip(cnn_filters, cnn_kernel_sizes)
        ])
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(sum(cnn_filters) + hidden_dim, 1)

    def forward(self, x):
        cnn_outputs = [F.relu(conv(x)) for conv in self.convs]
        cnn_outputs = [F.max_pool1d(output, output.size(2)).squeeze(2) for output in cnn_outputs]
        cnn_output = torch.cat(cnn_outputs, 1)

        lstm_output, _ = self.lstm(x)
        lstm_output = lstm_output[:, -1, :]  # Take the last output in the sequence

        x = torch.cat([cnn_output, lstm_output], 1)
        x = self.fc(x)
        return torch.sigmoid(x)
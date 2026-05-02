import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,     
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, 2)  # output: 2 classes (0, 1)

    def forward(self, x):
        # x shape: (batch, window, features)
        out, _ = self.lstm(x)

        
        out = out[:, -1, :]

        # classification
        out = self.fc(out)

        return out
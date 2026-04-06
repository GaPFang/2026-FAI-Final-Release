import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size=22, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, mask):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x - (1 - mask) * 1e9  # mask invalid (non-hand) cards
        return x  # returns logits; apply log_softmax / softmax outside

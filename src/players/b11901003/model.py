import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size=24, hidden_size=128, output_size=10):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x, mask):
        h = self.shared(x)
        logits = self.policy_head(h)
        logits = logits - (1 - mask) * 1e9   # mask invalid (non-hand) slots
        value = self.value_head(h).squeeze(-1)
        return logits, value                  # logits: raw; value: scalar

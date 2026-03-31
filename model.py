import torch
import torch.nn as nn

class SharedModel(nn.Module):
    def __init__(self, input_size):
        super(SharedModel, self).__init__()

        # Shared layers (Federated)
        self.shared = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Personalized layers (Local)
        self.personal = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

    def forward(self, x):
        x = self.shared(x)
        x = self.personal(x)
        return x
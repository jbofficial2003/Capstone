import torch
import torch.nn as nn


class SharedModel(nn.Module):

    def __init__(self, input_size, num_classes):
        super(SharedModel, self).__init__()

        # Reshape for CNN
        self.input_size = input_size

        # Adapter
        self.adapter = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # CNN Time-Series Layer
        self.shared = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.AdaptiveAvgPool1d(1)
        )

        # Personal Head
        self.personal = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):

        x = self.adapter(x)

        # reshape for CNN
        x = x.unsqueeze(1)

        x = self.shared(x)

        x = x.squeeze(-1)

        x = self.personal(x)

        return x
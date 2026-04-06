import torch.nn as nn


class SharedModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SharedModel, self).__init__()

        # 🔹 Stronger Adapter (local)
        self.adapter = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # 🔹 Improved Shared Backbone (federated)
        self.shared = nn.Sequential(
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # 🔹 Stronger Personal Head (local)
        self.personal = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.adapter(x)
        x = self.shared(x)
        x = self.personal(x)
        return x
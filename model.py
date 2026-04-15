import torch.nn as nn


class SharedModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SharedModel, self).__init__()

        # Adapter keeps client-specific feature shaping local.
        self.adapter = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),

            nn.Linear(64, 32),
            nn.GELU(),
        )

        # Shared backbone is kept normalization-stable for federated averaging.
        self.shared = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),

            nn.Linear(64, 32),
            nn.GELU(),
        )

        # Personal head stays local to each client.
        self.personal = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(32, 16),
            nn.GELU(),

            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.adapter(x)
        x = self.shared(x)
        x = self.personal(x)
        return x
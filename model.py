import torch.nn as nn


class SharedModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SharedModel, self).__init__()

        # Local input adapter (not federated): maps client-specific raw features
        # into a common latent representation.
        self.adapter = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Shared backbone (federated): same parameter shapes for all clients.
        self.shared = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Personalized head (not federated)
        self.personal = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.adapter(x)
        x = self.shared(x)
        x = self.personal(x)
        return x
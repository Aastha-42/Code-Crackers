import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

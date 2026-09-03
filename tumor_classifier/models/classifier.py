import torch.nn as nn


class Classifier(nn.Module):
    """Two-layer classifier head for AlexNet feature maps."""

    def __init__(self):
        super().__init__()
        self.name = "classifier"
        self.layer1 = nn.Linear(256 * 6 * 6, 64)
        self.layer2 = nn.Linear(64, 1)
        self.activation = nn.ReLU()

    def forward(self, img):
        flattened = img.view(-1, 256 * 6 * 6)
        hidden = self.activation(self.layer1(flattened))
        return self.layer2(hidden).squeeze(1)

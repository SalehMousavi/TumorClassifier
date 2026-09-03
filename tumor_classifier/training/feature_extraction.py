"""AlexNet feature extraction for transfer learning."""

import torch
import torchvision.models


def load_alexnet(device: torch.device):
    weights = torchvision.models.AlexNet_Weights.DEFAULT
    alexnet = torchvision.models.alexnet(weights=weights)
    alexnet = alexnet.to(device)
    alexnet.eval()
    return alexnet


def compute_alexnet_features(dataloader, alexnet, device: torch.device):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = alexnet.features(inputs)
            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

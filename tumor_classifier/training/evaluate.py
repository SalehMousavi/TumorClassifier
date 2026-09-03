"""Evaluation helpers for trained classifier checkpoints."""

from pathlib import Path

import torch
import torch.nn as nn

from tumor_classifier.config import TrainingConfig


def get_model_name(
    name,
    batch_size,
    learning_rate,
    epoch,
    checkpoint_dir: Path | None = None,
):
    checkpoint_dir = checkpoint_dir or TrainingConfig().checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(
        checkpoint_dir
        / f"model_{name}_bs{batch_size}_lr{learning_rate}_epoch{epoch}"
    )


def evaluate(net, loader, criterion, device: torch.device):
    net = net.to(device)
    net.eval()

    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i, (inputs, labels) in enumerate(loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels.float())
        corr = (outputs > 0.0).squeeze().long() != labels
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)

        predictions = (outputs > 0.0).squeeze().long()
        true_positives += int(((predictions == 1) & (labels == 1)).sum().item())
        false_positives += int(((predictions == 1) & (labels == 0)).sum().item())
        false_negatives += int(((predictions == 0) & (labels == 1)).sum().item())

    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return err, loss, f1_score, precision, recall

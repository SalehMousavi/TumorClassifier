"""Training loop and evaluation metrics for the classifier head."""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tumor_classifier.config import TrainingConfig
from tumor_classifier.training.evaluate import evaluate, get_model_name


def train_net(
    net,
    train_loader,
    val_loader,
    device: torch.device,
    config: TrainingConfig | None = None,
):
    config = config or TrainingConfig()
    net = net.to(device)
    torch.manual_seed(config.random_seed)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

    train_err = np.zeros(config.num_epochs)
    train_loss = np.zeros(config.num_epochs)
    val_err = np.zeros(config.num_epochs)
    val_loss = np.zeros(config.num_epochs)

    start_time = time.time()
    model_path = None

    for epoch in range(config.num_epochs):
        net.train()
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0

        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            corr = (outputs > 0.0).squeeze().long() != labels
            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i + 1)
        val_err[epoch], val_loss[epoch], _, _, _ = evaluate(
            net, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}: Train err: {train_err[epoch]}, Train loss: {train_loss[epoch]} | "
            f"Validation err: {val_err[epoch]}, Validation loss: {val_loss[epoch]}"
        )

        model_path = get_model_name(
            net.name,
            config.batch_size,
            config.learning_rate,
            epoch,
            checkpoint_dir=config.checkpoint_dir,
        )
        torch.save(net.state_dict(), model_path)

    print("Finished Training")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")

    np.savetxt(f"{model_path}_train_err.csv", train_err)
    np.savetxt(f"{model_path}_train_loss.csv", train_loss)
    np.savetxt(f"{model_path}_val_err.csv", val_err)
    np.savetxt(f"{model_path}_val_loss.csv", val_loss)

    return model_path

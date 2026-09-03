import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(path: str) -> None:
    """Plot training and validation error/loss from CSV files saved during training."""
    train_err = np.loadtxt(f"{path}_train_err.csv")
    val_err = np.loadtxt(f"{path}_val_err.csv")
    train_loss = np.loadtxt(f"{path}_train_loss.csv")
    val_loss = np.loadtxt(f"{path}_val_loss.csv")

    n = len(train_err)
    epochs = range(1, n + 1)

    plt.figure()
    plt.title("Train vs Validation Error")
    plt.plot(epochs, train_err, label="Train")
    plt.plot(epochs, val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc="best")
    plt.show()

    plt.figure()
    plt.title("Train vs Validation Loss")
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()

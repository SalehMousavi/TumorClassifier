"""Linear SVM baseline classifier on flattened pixel values."""

import numpy as np
import torch
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


class TumourClassifier:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.svm = svm.SVC(kernel="linear", probability=True, class_weight="balanced")
        self.scaler = StandardScaler()

    def train_svm(self, train_loader: DataLoader) -> None:
        images_list = []
        labels_list = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images_np = images.view(images.size(0), -1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            images_list.append(images_np)
            labels_list.append(labels_np)
            print(f"Loaded batch {batch_idx + 1}/{len(train_loader)}")

        all_images = np.vstack(images_list)
        all_labels = np.concatenate(labels_list)
        all_images = self.scaler.fit_transform(all_images)
        self.svm.fit(all_images, all_labels)

    def predict(self, data_loader: DataLoader):
        predictions = []
        labels = []

        for images, lbls in data_loader:
            images_np = images.view(images.size(0), -1).numpy()
            labels.extend(lbls.numpy())
            images_scaled = self.scaler.transform(images_np)
            predictions.extend(self.svm.predict(images_scaled))

        return np.array(predictions), np.array(labels)

    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        predictions, labels = self.predict(data_loader)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted"),
            "f1": f1_score(labels, predictions, average="weighted"),
        }

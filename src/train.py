"""Train LeNet-5 on MNIST with optional simulated annealing learning-rate search."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms


class LeNet5(nn.Module):
    """LeNet-5 style convolutional neural network for 28x28 grayscale images."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv_dataset(csv_path: Path) -> TensorDataset:
    """Load MNIST-format CSV with labels in the first column and 784 pixel columns."""
    data = pd.read_csv(csv_path)
    images = data.iloc[:, 1:].values.reshape(-1, 1, 28, 28).astype("float32") / 255.0
    labels = data.iloc[:, 0].values.astype("int64")
    return TensorDataset(torch.tensor(images), torch.tensor(labels))


def get_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if args.data_source == "csv":
        if args.train_csv is None or args.test_csv is None:
            raise ValueError("CSV mode requires both --train-csv and --test-csv.")
        full_train = load_csv_dataset(Path(args.train_csv))
        test_dataset = load_csv_dataset(Path(args.test_csv))
    else:
        transform = transforms.ToTensor()
        full_train = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    val_size = int(args.validation_split * len(full_train))
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> list[float]:
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] - loss: {avg_loss:.4f}")
    return losses


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def simulated_annealing_lr_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    initial_lr: float,
    initial_temp: float,
    cooling_rate: float,
    epochs_per_trial: int,
) -> tuple[float, dict[str, list[float]]]:
    current_lr = initial_lr
    current_temp = initial_temp
    best_lr = current_lr
    best_val_accuracy = 0.0

    history = {
        "temperature": [],
        "learning_rate": [],
        "validation_accuracy": [],
        "test_accuracy": [],
    }

    while current_temp > 1e-3:
        model = LeNet5().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=current_lr)
        train_one_model(model, train_loader, criterion, optimizer, device, epochs_per_trial)

        val_accuracy = evaluate(model, val_loader, device)
        test_accuracy = evaluate(model, test_loader, device)

        history["temperature"].append(current_temp)
        history["learning_rate"].append(current_lr)
        history["validation_accuracy"].append(val_accuracy)
        history["test_accuracy"].append(test_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_lr = current_lr

        candidate_lr = max(current_lr + np.random.uniform(-0.001, 0.001), 1e-6)
        accept_probability = np.exp((val_accuracy - best_val_accuracy) / current_temp)
        if val_accuracy >= best_val_accuracy or accept_probability > random.random():
            current_lr = candidate_lr

        current_temp *= cooling_rate
        print(
            f"temperature={current_temp:.4f}, lr={current_lr:.6f}, "
            f"val_acc={val_accuracy:.2f}%, test_acc={test_accuracy:.2f}%"
        )

    return best_lr, history


def save_line_plot(values: list[float], title: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(values) + 1), values)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LeNet-5 on MNIST.")
    parser.add_argument("--data-source", choices=["torchvision", "csv"], default="torchvision")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimize-lr", action="store_true")
    parser.add_argument("--initial-temperature", type=float, default=10.0)
    parser.add_argument("--cooling-rate", type=float, default=0.95)
    parser.add_argument("--epochs-per-trial", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(args)
    lr = args.learning_rate

    if args.optimize_lr:
        lr, history = simulated_annealing_lr_search(
            train_loader,
            val_loader,
            test_loader,
            device,
            initial_lr=args.learning_rate,
            initial_temp=args.initial_temperature,
            cooling_rate=args.cooling_rate,
            epochs_per_trial=args.epochs_per_trial,
        )
        print(f"Best learning rate from simulated annealing: {lr:.6f}")
        pd.DataFrame(history).to_csv(Path(args.output_dir) / "annealing_history.csv", index=False)

    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = train_one_model(model, train_loader, criterion, optimizer, device, args.epochs)

    val_accuracy = evaluate(model, val_loader, device)
    test_accuracy = evaluate(model, test_loader, device)
    print(f"Validation accuracy: {val_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "lenet5_mnist.pt")
    save_line_plot(losses, "Training Loss vs. Epoch", "Loss", output_dir / "training_loss.png")


if __name__ == "__main__":
    main()

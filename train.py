import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from models import EnhancedNNClassifier
from parameters import *
from matplotlib import pyplot as plt
import os

# ! Handle this later to read the several csv data files

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data: pd.DataFrame, batch_size: int) -> tuple:
    """
    This function loads the data and returns the features and target
    """
    # Split the data into features and target
    X = data.drop("Species_Population", axis=1).values
    Y = data["Species_Population"].values

    # standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, Y, test_size=0.3, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(2 / 3), random_state=RANDOM_SEED
    )

    # convert arrays to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    return (
        train_loader,
        test_loader,
        val_loader,
    )


def train_model(num_epochs, model, criterion, optimizer, train_loader, val_loader):
    model.train()
    epoch_losses = []
    val_accuracies = []

    if len(train_loader) == 0:
        raise ValueError(
            "Training loader has no batches. Please check your dataset and batch size."
        )

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_losses.append(avg_loss)

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total if total > 0 else 0
        val_accuracies.append(val_accuracy)
        model.train()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    print("Finished Training")

    # Save the model weights
    model_path = os.path.join("models", "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot training loss and validation accuracy
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(epoch_losses, label="Training Loss")
    # plt.title("Loss Over Epochs")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(val_accuracies, label="Validation Accuracy", color="r")
    # plt.title("Validation Accuracy Over Epochs")
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.grid(True)

    # plt.show()


def evaluate_model(model, loader):
    model.eval()  # Evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}")
    return accuracy


# data = pd.read_csv("Cleaned_data.csv")

# model = EnhancedNNClassifier(input_size=14, num_classes=10).to(device)

# train_loader, test_loader, val_loader = load_data(data, batch_size=64)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# train_model(
#     num_epochs=2500,
#     model=model,
#     criterion=nn.CrossEntropyLoss(),
#     optimizer=optimizer,
#     train_loader=train_loader,
#     val_loader=val_loader,
# )

# evaluate_model(model, test_loader)

import torch
import torch.nn as nn
import pandas as pd
from train import train_model, load_data, evaluate_model
import os
from parameters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HyperparamaterModelTester:
    """
    This class is used to test the model with different hyperparameters
    """

    def __init__(
        self,
        models: nn.Module,
        learning_rates: list[float],
        batch_sizes: list[int],
        optimizers: list[torch.optim.Optimizer],
        criterion: nn.Module,
        data: pd.DataFrame,
        num_epochs: int = 100,
    ):
        self.models = models
        self.learning_rates = learning_rates
        self.batch_sizes = batch_sizes
        self.optimizers = optimizers
        self.criterion = criterion
        self.epochs = num_epochs
        self.data = data

        self.accuracies = []

    def test(self):
        """
        This method is used to test the model with different hyperparameters
        """
        for learning_rate in self.learning_rates:
            for batch_size in self.batch_sizes:
                for optimizer_class in self.optimizers:
                    for model in self.models:
                        model_instance = model.to(device)
                        train_loader, test_loader, val_loader = load_data(
                            self.data, batch_size
                        )
                        optimizer = optimizer_class(
                            model_instance.parameters(), lr=learning_rate
                        )

                        train_model(
                            self.epochs,
                            model_instance,
                            self.criterion,
                            optimizer,
                            train_loader,
                            val_loader,
                        )
                        self.accuracies.append(
                            f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer}, Model: {model.__class__.__name__}, Accuracy: {evaluate_model(model, test_loader)}",
                        )

    def get_accuracies(self):
        """
        This method is used to get the accuracies of the models
        """
        for accuracy in self.accuracies:
            print(accuracy)


class NoisyDataModelTest:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer_class: torch.optim.Optimizer,
        optimizer_args: dict,
        batch_size: int,
        num_epochs: int = 100,
        data_dir: str = "processed_data/",
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_args = optimizer_args
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.data_dir = data_dir
        self.accuracies = []

    def test(self):
        # Iterate through each CSV file in the directory
        for file_name in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file_name)
            if file_path.endswith(".csv"):
                data = pd.read_csv(file_path)
                train_loader, test_loader, val_loader = load_data(data, self.batch_size)

                # Initialize the optimizer with model parameters
                optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_args
                )

                # Train the model
                train_model(
                    self.num_epochs,
                    self.model,
                    self.criterion,
                    optimizer,
                    train_loader,
                    val_loader,
                )

                # Evaluate the model
                accuracy = evaluate_model(self.model, test_loader)
                print(f"Processed {file_name}: Accuracy = {accuracy:.4f}")
                self.accuracies.append((file_name, accuracy))

    def get_accuracies(self):
        # Return a sorted list of accuracies
        return sorted(self.accuracies, key=lambda x: x[1], reverse=True)


# Example of using the modified class
if __name__ == "__main__":
    from models import EnhancedNNClassifier, SimpleNNClassifier
    from torch.optim import SGD
    from torch.nn import CrossEntropyLoss

    model = EnhancedNNClassifier(input_size=15, num_classes=10)
    criterion = CrossEntropyLoss()
    optimizer_args = {"lr": 0.001}  # Dictionary of arguments for the optimizer

    tester = NoisyDataModelTest(
        model=model,
        criterion=criterion,
        optimizer_class=SGD,
        optimizer_args=optimizer_args,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        data_dir="processed_data/",
    )

    tester.test()
    print(tester.get_accuracies())

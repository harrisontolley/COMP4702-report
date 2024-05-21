import torch.nn as nn
import torch.nn.functional as F


class SimpleNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super(SimpleNNClassifier, self).__init__()
        # Define the first hidden layer
        self.layer1 = nn.Linear(input_size, 512)  # First hidden layer with 128 nodes
        # Define the second hidden layer
        self.layer2 = nn.Linear(512, 256)  # Second hidden layer with 64 nodes
        # Output layer that will output probabilities for the 10 classes
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        # Pass the input through the first hidden layer and apply a ReLU activation function
        x = F.relu(self.layer1(x))
        # Pass the result through the second hidden layer and apply another ReLU
        x = F.relu(self.layer2(x))
        # Pass the result through the output layer
        x = self.output(x)
        return x


class EnhancedNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super(EnhancedNNClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)  # Increased complexity
        self.batch_norm1 = nn.BatchNorm1d(256)  # Batch normalization
        self.dropout1 = nn.Dropout(0.5)  # Dropout
        self.layer2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout1(x)
        x = F.relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout2(x)
        x = self.output(x)
        return x

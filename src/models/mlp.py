import sys
import os

# Ensure correct module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBase(nn.Module):
    """
    A basic MLP module used as a building block for the three MLP variants.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        """
        Args:
            input_dim (int): Dimensionality of input.
            hidden_dim (int): Dimensionality of hidden layers.
            output_dim (int): Dimensionality of output.
            num_layers (int): Total number of layers (including input and output layers).
        """
        super(MLPBase, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Hidden layers (if any)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPForecast(nn.Module):
    """
    MLP for forecasting tasks.
    Accepts input of shape (batch, T1, D1) and outputs a tensor of shape (batch, T2, D2).
    """

    def __init__(self, D1, T1, D2, T2, hidden_dim=512, num_layers=2):
        """
        Args:
            D1 (int): Number of input features.
            T1 (int): Number of input time steps.
            D2 (int): Number of output features.
            T2 (int): Number of output time steps.
            hidden_dim (int): Hidden layer size.
            num_layers (int): Total number of layers in the MLP.
        """
        super(MLPForecast, self).__init__()
        self.D1 = D1
        self.T1 = T1
        self.D2 = D2
        self.T2 = T2
        # Flattened input dimension is T1 * D1, output is T2 * D2
        input_dim = T1 * D1
        output_dim = T2 * D2
        self.mlp = MLPBase(input_dim, hidden_dim, output_dim, num_layers)

    def forward(self, x):
        # x shape: (batch, T1, D1)
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        out = self.mlp(x_flat)
        # Reshape output to (batch, T2, D2)
        out = out.view(batch_size, self.T2, self.D2)
        return out


class MLPRegression(nn.Module):
    """
    MLP for regression tasks.
    Expects an input of shape (batch, T1, D1) and outputs a single number per sample.
    """

    def __init__(self, D1, T1, hidden_dim=512, num_layers=2):
        """
        Args:
            D1 (int): Number of input features.
            T1 (int): Number of time steps.
            hidden_dim (int): Hidden layer size.
            num_layers (int): Total number of layers in the MLP.
        """
        super(MLPRegression, self).__init__()
        self.T1 = T1
        self.D1 = D1
        input_dim = T1 * D1
        self.mlp = MLPBase(input_dim, hidden_dim, 1, num_layers)

    def forward(self, x):
        # x shape: (batch, T1, D1)
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        out = self.mlp(x_flat)
        # Output shape: (batch, 1)
        return out


class MLPClassification(nn.Module):
    """
    MLP for classification tasks.
    Expects an input of shape (batch, T1, D1) and outputs a probability between 0 and 1.
    """

    def __init__(self, D1, T1, hidden_dim=512, num_layers=2):
        """
        Args:
            D1 (int): Number of input features.
            T1 (int): Number of time steps.
            hidden_dim (int): Hidden layer size.
            num_layers (int): Total number of layers in the MLP.
        """
        super(MLPClassification, self).__init__()
        self.T1 = T1
        self.D1 = D1
        input_dim = T1 * D1
        self.mlp = MLPBase(input_dim, hidden_dim, 1, num_layers)

    def forward(self, x):
        # x shape: (batch, T1, D1)
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        out = self.mlp(x_flat)
        # Apply sigmoid to output a probability between 0 and 1
        # prob = torch.sigmoid(out)
        # Output shape: (batch, 1)
        return out


class MLPMultiLabelClassification(nn.Module):
    """
    MLP for multi-label classification.
    Expects input of shape (batch, T1, D1) and outputs (batch, num_labels) probabilities in (0,1)
    unless return_logits=True (then raw logits are returned).
    """
    def __init__(self, D1, T1, num_labels, hidden_dim=512, num_layers=2, return_logits=False):
        """
        Args:
            D1 (int): Number of input features.
            T1 (int): Number of time steps.
            num_labels (int): Number of independent labels (output dimensions).
            hidden_dim (int): Hidden layer size for MLPBase.
            num_layers (int): Total number of layers in MLPBase.
            return_logits (bool): If True, return raw logits (use with BCEWithLogitsLoss).
        """
        super().__init__()
        self.T1 = T1
        self.D1 = D1
        self.num_labels = num_labels
        self.return_logits = return_logits

        input_dim = T1 * D1
        self.mlp = MLPBase(input_dim, hidden_dim, num_labels, num_layers)

    def forward(self, x):
        # x: (batch, T1, D1)
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        logits = self.mlp(x_flat)
        # return torch.sigmoid(logits)
        return logits


#######################################
#             Main Script
#######################################
if __name__ == "__main__":
    # Define input and output dimensions/time steps
    D1, T1, D2, T2 = 66, 1, 40, 24
    batch_size = 4

    # --- Forecasting Test ---
    # Input: (batch, T1, D1), Output: (batch, T2, D2)
    x_forecast = torch.randn(batch_size, T1, D1)
    forecast_model = MLPForecast(D1, T1, D2, T2)
    forecast_output = forecast_model(x_forecast)
    print("MLP Forecast output shape:", forecast_output.shape)
    # Expected: (batch_size, T2, D2)

    # --- Regression Test ---
    # Input: (batch, T1, D1), Output: (batch, 1)
    x_regression = torch.randn(batch_size, T1, D1)
    regression_model = MLPRegression(D1, T1)
    regression_output = regression_model(x_regression)
    print("MLP Regression output shape:", regression_output.shape)
    # Expected: (batch_size, 1)

    # --- Classification Test ---
    # Input: (batch, T1, D1), Output: (batch, 1) with values in [0, 1]
    x_classification = torch.randn(batch_size, T1, D1)
    classification_model = MLPClassification(D1, T1)
    classification_output = classification_model(x_classification)
    print("MLP Classification output shape:", classification_output.shape)
    # Expected: (batch_size, 1)

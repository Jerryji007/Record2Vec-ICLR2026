import torch
import torch.nn as nn


#######################################
#         LSTM Forecasting
#######################################
class LSTMForecast(nn.Module):
    """
    LSTM for forecasting tasks.
    Accepts input of shape (batch, T1, D1) and outputs a tensor of shape (batch, T2, D2).
    Uses an encoder-decoder architecture.
    """

    def __init__(self, D1, T1, D2, T2, hidden_dim=512, num_layers=1):
        """
        Args:
            D1 (int): Number of input features.
            T1 (int): Number of input time steps.
            D2 (int): Number of output features.
            T2 (int): Number of output time steps.
            hidden_dim (int): Hidden layer size.
            num_layers (int): Number of layers in the LSTM.
        """
        super(LSTMForecast, self).__init__()
        self.D1 = D1
        self.T1 = T1
        self.D2 = D2
        self.T2 = T2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder processes input sequence of shape (batch, T1, D1)
        self.encoder = nn.LSTM(
            input_size=D1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        # Decoder generates T2 time steps; its input size is D2.
        self.decoder = nn.LSTM(
            input_size=D2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        # Final linear layer maps from hidden_dim to output feature dimension D2.
        self.fc = nn.Linear(hidden_dim, D2)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, T1, D1)
        Returns:
            torch.Tensor: Output tensor of shape (batch, T2, D2)
        """
        batch_size = x.size(0)
        # Encoder: process the entire input sequence.
        _, (h, c) = self.encoder(x)

        # Decoder: use a start token (zeros) for the first time step with dimension D2.
        decoder_input = torch.zeros(batch_size, 1, self.D2, device=x.device)
        outputs = []
        decoder_hidden = (h, c)
        for _ in range(self.T2):
            # Produce one output time step.
            out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # Map LSTM output to the feature dimension D2.
            pred = self.fc(out.squeeze(1))  # shape: (batch, D2)
            outputs.append(pred.unsqueeze(1))  # shape: (batch, 1, D2)
            # Use prediction as the next input.
            decoder_input = pred.unsqueeze(1)

        # Concatenate outputs along the time dimension: (batch, T2, D2)
        outputs = torch.cat(outputs, dim=1)
        return outputs


#######################################
#         LSTM Regression
#######################################
class LSTMRegression(nn.Module):
    """
    LSTM for regression tasks.
    Accepts input of shape (batch, T1, D1) and outputs a single number per sample (batch, 1).
    """

    def __init__(self, D1, T1, hidden_dim=512, num_layers=1):
        """
        Args:
            D1 (int): Number of input features.
            T1 (int): Number of time steps.
            hidden_dim (int): Hidden layer size.
            num_layers (int): Number of layers in the LSTM.
        """
        super(LSTMRegression, self).__init__()
        self.lstm = nn.LSTM(
            input_size=D1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, T1, D1)
        Returns:
            torch.Tensor: Output tensor of shape (batch, 1)
        """
        # Process the input sequence (batch, T1, D1)
        _, (h, _) = self.lstm(x)
        h_last = h[-1]  # (batch, hidden_dim)
        out = self.fc(h_last)  # (batch, 1)
        return out


#######################################
#         LSTM Classification
#######################################
class LSTMClassification(nn.Module):
    """
    LSTM for classification tasks.
    Accepts input of shape (batch, T1, D1) and outputs a probability between 0 and 1 (batch, 1).
    """

    def __init__(self, D1, T1, hidden_dim=512, num_layers=1):
        """
        Args:
            D1 (int): Number of input features.
            T1 (int): Number of time steps.
            hidden_dim (int): Hidden layer size.
            num_layers (int): Number of layers in the LSTM.
        """
        super(LSTMClassification, self).__init__()
        self.lstm = nn.LSTM(
            input_size=D1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, T1, D1)
        Returns:
            torch.Tensor: Output tensor of shape (batch, 1) with probabilities in (0,1)
        """
        _, (h, _) = self.lstm(x)
        h_last = h[-1]
        out = self.fc(h_last)
        # prob = torch.sigmoid(out)
        return out


#######################################
#         LSTM Multi-Label Classification
#######################################
class LSTMMultiLabelClassification(nn.Module):
    """
    LSTM for multi-label classification.
    Input:  (batch, T1, D1)
    Output: (batch, num_labels) with probabilities in (0, 1)
    """
    def __init__(
        self,
        D1: int,
        T1: int,
        num_labels: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        return_logits: bool = False,
    ):
        super().__init__()
        self.return_logits = return_logits
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=D1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor (batch, T1, D1)
        returns:
          - logits if return_logits=True (batch, num_labels)
          - probabilities (sigmoid) otherwise
        """
        _, (h, _) = self.lstm(x)

        if self.bidirectional:
            h_last = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            h_last = h[-1]

        logits = self.fc(h_last)
        # if self.return_logits:
        #     return logits
        # return torch.sigmoid(logits)
        return logits


#######################################
#             Main Test Script
#######################################
if __name__ == "__main__":
    # Define parameters
    D1, T1, D2, T2 = 71, 5, 55, 24
    hidden_dim = 128
    num_layers = 1
    batch_size = 256

    # --- LSTM Forecasting Test ---
    # Input shape: (batch, T1, D1), Output shape: (batch, T2, D2)
    x_forecast = torch.randn(batch_size, T1, D1)
    lstm_forecast = LSTMForecast(D1, T1, D2, T2, hidden_dim, num_layers)
    forecast_output = lstm_forecast(x_forecast)
    print("LSTM Forecast output shape:", forecast_output.shape)
    # Expected: (batch_size, T2, D2)

    # --- LSTM Regression Test ---
    # Input shape: (batch, T1, D1), Output shape: (batch, 1)
    x_regression = torch.randn(batch_size, T1, D1)
    lstm_regression = LSTMRegression(D1, T1, hidden_dim, num_layers)
    regression_output = lstm_regression(x_regression)
    print("LSTM Regression output shape:", regression_output.shape)
    # Expected: (batch_size, 1)

    # --- LSTM Classification Test ---
    # Input shape: (batch, T1, D1), Output shape: (batch, 1)
    x_classification = torch.randn(batch_size, T1, D1)
    lstm_classification = LSTMClassification(D1, T1, hidden_dim, num_layers)
    classification_output = lstm_classification(x_classification)
    print("LSTM Classification output shape:", classification_output.shape)
    # Expected: (batch_size, 1) with values in (0,1)

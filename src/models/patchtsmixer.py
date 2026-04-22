import torch
import torch.nn as nn

# ---------------------------
# Helper Modules
# ---------------------------


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        RevIN module for reversible instance normalization.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        # Compute mean and stdev across all dims except batch and channel
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class PatchMixerLayer(nn.Module):
    def __init__(self, dim, a, kernel_size=8):
        """
        A single patch-mixing layer.
        """
        super(PatchMixerLayer, self).__init__()
        self.Resnet = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm1d(dim),
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size=1), nn.GELU(), nn.BatchNorm1d(a)
        )

    def forward(self, x):
        x = x + self.Resnet(x)  # Residual connection
        x = self.Conv_1x1(x)
        return x


# ---------------------------
# Base Class for PatchTSMixer variants
# ---------------------------
class PatchTSMixerBase(nn.Module):
    def __init__(
        self,
        D1,
        T1,
        hidden_dim=256,
        dropout=0.1,
        layer_dim=1,
        patch_size=1,
        stride=5,
        kernel_size=3,
        revin=True,
        affine=True,
        subtract_last=False,
    ):
        """
        Base module for PatchTSMixer models.
        Args:
            D1 (int): Number of input features (channels).
            T1 (int): Number of input time steps.
            hidden_dim (int): Hidden dimension (d_model).
            dropout (float): Dropout probability.
            layer_dim (int): Number of patch-mixing layers.
            patch_size (int): Size of each patch.
            stride (int): Stride for unfolding patches.
            kernel_size (int): Kernel size used in PatchMixerLayer.
            revin (bool): Whether to apply RevIN.
            affine (bool): RevIN affine flag.
            subtract_last (bool): RevIN subtract_last flag.
        """
        super(PatchTSMixerBase, self).__init__()
        self.D1 = D1
        self.T1 = T1
        # For these models, we assume input shape is (batch, T1, D1)
        # and later we permute to (batch, D1, T1)
        self.patch_size = patch_size
        self.stride = stride
        self.kernel_size = kernel_size
        # Calculate number of patches from the time dimension.
        self.patch_num = int((T1 - patch_size) / stride + 1) + 1
        self.a = self.patch_num  # used in the PatchMixerLayer

        self.d_model = hidden_dim  # projection dimension

        # Padding layer to ensure we have enough patches
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        # Linear projection applied to each patch (patch_size -> d_model)
        self.W_P = nn.Linear(patch_size, self.d_model)
        # Dropout layer (applied after projection)
        self.dropout_layer = nn.Dropout(dropout)

        # Build a stack of PatchMixer layers.
        self.PatchMixer_blocks = nn.ModuleList(
            [
                PatchMixerLayer(
                    dim=self.patch_num, a=self.a, kernel_size=self.kernel_size
                )
                for _ in range(layer_dim)
            ]
        )

        # RevIN layer for reversible normalization on the input features
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(D1, affine=affine, subtract_last=subtract_last)

    def forward_features(self, x):
        """
        Extract features using patch unfolding, projection, and dropout.
        Args:
            x: tensor of shape (batch, T1, D1)
        Returns:
            x: patch features of shape (batch, D1, patch_num, d_model)
            bs: original batch size
            nvars: number of input features (D1)
        """
        bs = x.size(0)
        nvars = x.size(-1)  # equals D1
        if self.revin:
            x = self.revin_layer(x, "norm")
        # Permute to (batch, D1, T1)
        x = x.permute(0, 2, 1)
        # Apply padding and then unfold to form patches along the time dimension.
        x_lookback = self.padding_patch_layer(x)
        # x becomes (batch, D1, patch_num, patch_size)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # Apply linear projection on the patch dimension: now shape -> (batch, D1, patch_num, d_model)
        x = self.W_P(x)
        x = self.dropout_layer(x)
        return x, bs, nvars


# ---------------------------
# Forecasting Variant
# ---------------------------
class PatchTSMixerForecast(PatchTSMixerBase):
    def __init__(
        self,
        D1,
        T1,
        D2,
        T2,
        hidden_dim=256,
        dropout=0.1,
        layer_dim=1,
        patch_size=3,
        stride=5,
        kernel_size=3,
        revin=True,
        affine=True,
        subtract_last=False,
    ):
        """
        PatchTSMixer for forecasting.
        Args:
            D1 (int): Number of input features.
            T1 (int): Number of input time steps.
            D2 (int): Number of output features.
            T2 (int): Number of forecast (output) time steps.
            (other args as in base)
        """
        super(PatchTSMixerForecast, self).__init__(
            D1,
            T1,
            hidden_dim,
            dropout,
            layer_dim,
            patch_size,
            stride,
            kernel_size,
            revin,
            affine,
            subtract_last,
        )
        self.T2 = T2
        self.D2 = D2
        # Head definitions now operate on the full tensor.
        # Flatten all dimensions except batch: (D1 * patch_num * d_model)
        in_features = D1 * self.patch_num * self.d_model
        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features, T2 * D2),
            nn.Dropout(dropout),
        )
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features, T2 * D2 * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(T2 * D2 * 2, T2 * D2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, T1, D1)
        Returns:
            Tensor of shape (batch, T2, D2)
        """
        x, bs, _ = self.forward_features(x)  # x shape: (bs, D1, patch_num, d_model)
        # Compute head0 from the original patch features.
        u = self.head0(x)
        # Pass through the patch mixer layers.
        for block in self.PatchMixer_blocks:
            bsD, patch_num, d_model = x.shape[0] * x.shape[1], x.shape[2], x.shape[3]
            x_reshaped = x.reshape(bs * self.D1, self.patch_num, self.d_model)
            x_reshaped = block(x_reshaped)
            x = x_reshaped.reshape(bs, self.D1, self.patch_num, self.d_model)
        v = self.head1(x)
        out = u + v  # residual connection between heads
        # Reshape to (batch, T2, D2)
        out = out.view(bs, self.T2, self.D2)
        # Apply denormalization only if input and output features match.
        if self.revin:
            if self.D1 == self.D2:
                out = self.revin_layer(out, "denorm")
            # else:
            #     print(
            #         "Warning: Skipping RevIN denorm because D1 != D2 ({} vs {}).".format(
            #             self.D1, self.D2
            #         )
            #     )
        return out


# ---------------------------
# Regression Variant
# ---------------------------
class PatchTSMixerRegression(PatchTSMixerBase):
    def __init__(
        self,
        D1,
        T1,
        hidden_dim=256,
        dropout=0.1,
        layer_dim=1,
        patch_size=3,
        stride=3,
        kernel_size=3,
        revin=True,
        affine=True,
        subtract_last=False,
    ):
        """
        PatchTSMixer for regression.
        Args:
            D1 (int): Number of input features.
            T1 (int): Number of input time steps.
            (other args as in base)
        """
        super(PatchTSMixerRegression, self).__init__(
            D1,
            T1,
            hidden_dim,
            dropout,
            layer_dim,
            patch_size,
            stride,
            kernel_size,
            revin,
            affine,
            subtract_last,
        )
        in_features = D1 * self.patch_num * self.d_model
        # Regression head: map flattened patch features to a single output.
        self.regression_head = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, T1, D1)
        Returns:
            Tensor of shape (batch, 1)
        """
        x, bs, _ = self.forward_features(x)
        for block in self.PatchMixer_blocks:
            # Process along the patch dimension.
            bsD, patch_num, d_model = x.shape[0] * x.shape[1], x.shape[2], x.shape[3]
            x_reshaped = x.reshape(bs * self.D1, self.patch_num, self.d_model)
            x_reshaped = block(x_reshaped)
            x = x_reshaped.reshape(bs, self.D1, self.patch_num, self.d_model)
        out = self.regression_head(x)
        return out


# ---------------------------
# Classification Variant
# ---------------------------
class PatchTSMixerClassification(PatchTSMixerBase):
    def __init__(
        self,
        D1,
        T1,
        hidden_dim=256,
        dropout=0.1,
        layer_dim=1,
        patch_size=3,
        stride=3,
        kernel_size=3,
        revin=True,
        affine=True,
        subtract_last=False,
    ):
        """
        PatchTSMixer for classification.
        Args:
            D1 (int): Number of input features.
            T1 (int): Number of input time steps.
            (other args as in base)
        """
        super(PatchTSMixerClassification, self).__init__(
            D1,
            T1,
            hidden_dim,
            dropout,
            layer_dim,
            patch_size,
            stride,
            kernel_size,
            revin,
            affine,
            subtract_last,
        )
        in_features = D1 * self.patch_num * self.d_model
        # Classification head: map flattened patch features to a probability.
        self.classification_head = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(in_features, 1), nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, T1, D1)
        Returns:
            Tensor of shape (batch, 1)
        """
        x, bs, _ = self.forward_features(x)
        for block in self.PatchMixer_blocks:
            bsD, patch_num, d_model = x.shape[0] * x.shape[1], x.shape[2], x.shape[3]
            x_reshaped = x.reshape(bs * self.D1, self.patch_num, self.d_model)
            # print(x_reshaped.shape)
            x_reshaped = block(x_reshaped)
            x = x_reshaped.reshape(bs, self.D1, self.patch_num, self.d_model)
        out = self.classification_head(x)
        return out

# ---------------------------
# Multi-Label Classification
# ---------------------------
class PatchTSMixerMultiLabelClassification(PatchTSMixerBase):
    def __init__(
        self,
        D1,
        T1,
        num_labels,
        hidden_dim=256,
        dropout=0.1,
        layer_dim=1,
        patch_size=3,
        stride=3,
        kernel_size=3,
        revin=True,
        affine=True,
        subtract_last=False,
        return_logits: bool = False,
    ):
        """
        PatchTSMixer for multi-label classification.

        Args:
            D1 (int): Number of input features.
            T1 (int): Number of input time steps.
            num_labels (int): Number of independent labels to predict.
            return_logits (bool): If True, output raw logits (use with BCEWithLogitsLoss).
                                  If False, output probabilities via sigmoid.
            (other args match PatchTSMixerBase)
        """
        super().__init__(
            D1,
            T1,
            hidden_dim,
            dropout,
            layer_dim,
            patch_size,
            stride,
            kernel_size,
            revin,
            affine,
            subtract_last,
        )
        self.return_logits = return_logits

        in_features = D1 * self.patch_num * self.d_model
        # Head maps flattened patch features -> num_labels
        self.classification_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features, num_labels),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, T1, D1)
        Returns:
            Tensor of shape (batch, num_labels)
                - logits if return_logits=True
                - probabilities in (0,1) otherwise
        """
        x, bs, _ = self.forward_features(x)  # x: (bs, D1, patch_num, d_model)

        # pass through PatchMixer blocks
        for block in self.PatchMixer_blocks:
            # reshape to (bs*D1, patch_num, d_model), apply block, reshape back
            x_reshaped = x.reshape(bs * self.D1, self.patch_num, self.d_model)
            x_reshaped = block(x_reshaped)
            x = x_reshaped.reshape(bs, self.D1, self.patch_num, self.d_model)

        logits = self.classification_head(x)  # (bs, num_labels)
        # if self.return_logits:
        #     return logits
        # return torch.sigmoid(logits)
        return logits


# ---------------------------
# Main Test Script
# ---------------------------
if __name__ == "__main__":
    # Assume input shape: (batch, T1, D1)
    batch = 4096
    T1 = 8  # input time steps
    D1 = 1000  # number of input features
    T2 = 24  # forecast time steps for forecasting variant
    D2 = 40  # number of output features for forecasting variant

    x = torch.randn(batch, T1, D1)

    # Forecasting test
    model_forecast = PatchTSMixerForecast(
        D1=D1, T1=T1, D2=D2, T2=T2, hidden_dim=128, dropout=0.1, layer_dim=2
    )
    out_forecast = model_forecast(x)
    print("Forecasting output shape:", out_forecast.shape)  # Expected: (batch, T2, D2)

    # Regression test
    model_regression = PatchTSMixerRegression(
        D1=D1, T1=T1, hidden_dim=128, dropout=0.1, layer_dim=2
    )
    out_regression = model_regression(x)
    print("Regression output shape:", out_regression.shape)  # Expected: (batch, 1)

    # Classification test
    model_classification = PatchTSMixerClassification(
        D1=D1, T1=T1, hidden_dim=128, dropout=0.1, layer_dim=2
    )
    out_classification = model_classification(x)
    print(
        "Classification output shape:", out_classification.shape
    )  # Expected: (batch, 1)

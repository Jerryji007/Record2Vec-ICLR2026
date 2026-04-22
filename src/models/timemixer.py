import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Ensure correct module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.layers.Autoformer_EncDec import series_decomp
from src.models.layers.Embed import DataEmbedding_wo_pos
from src.models.layers.StandardNorm import Normalize
from src.models.layers.timemixerlayer import PastDecomposableMixing


#######################################
#  Default Configuration Class for Forecasting
#######################################
class ConfigForecast:
    def __init__(self, D1, T1, D2, T2):
        self.task_name = "long_term_forecast"
        self.enc_in = D1  # number of input features
        self.seq_len = T1  # input time steps
        self.pred_len = T2  # forecast time steps
        self.dec_in = D2  # decoder input channels (for forecasting)
        self.c_out = D2  # forecast output channels
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 2048
        self.moving_avg = 12
        self.factor = 1
        self.distil = True
        self.dropout = 0.1
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.channel_independence = 0
        self.decomp_method = "dft_decomp"
        self.top_k = 5
        self.use_norm = 1
        self.down_sampling_layers = 2
        self.down_sampling_window = 2
        self.down_sampling_method = "avg"
        self.use_future_temporal_feature = 0
        self.freq = "h"
        self.label_len = T1  # usually the input length


#######################################
#         Base Class
#######################################
class TimeMixerBase(nn.Module):
    def __init__(self, config):
        super(TimeMixerBase, self).__init__()
        self.config = config
        self.enc_in = config.enc_in
        self.seq_len = config.seq_len
        self.down_sampling_layers = config.down_sampling_layers
        self.down_sampling_window = config.down_sampling_window
        self.down_sampling_method = config.down_sampling_method
        self.channel_independence = config.channel_independence

        # If not channel independent, apply series decomposition.
        if self.channel_independence == 0:
            self.preprocess = series_decomp(config.moving_avg)

        # Build the embedding layer.
        in_channels = config.enc_in if self.channel_independence == 0 else 1
        self.enc_embedding = DataEmbedding_wo_pos(
            in_channels, config.d_model, config.embed, config.freq, config.dropout
        )

        # Build PastDecomposableMixing blocks.
        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(config) for _ in range(config.e_layers)]
        )

        # Build normalization layers for each scale.
        self.normalize_layers = nn.ModuleList(
            [
                Normalize(config.enc_in, affine=True, non_norm=(config.use_norm == 0))
                for _ in range(self.down_sampling_layers + 1)
            ]
        )

    def _multi_scale_process_inputs(self, x):
        """
        Apply multi-scale pooling to x.
        x: tensor of shape [B, T, D]
        Returns a list of tensors (each scale) with shape [B, T_scale, D]
        """
        if self.down_sampling_method == "max":
            down_pool = nn.MaxPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "avg":
            down_pool = nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            return [x]

        # Permute to [B, D, T] for pooling.
        x = x.permute(0, 2, 1)
        x_ori = x
        x_list = [x.permute(0, 2, 1)]  # original scale: [B, T, D]
        for _ in range(self.down_sampling_layers):
            x_sample = down_pool(x_ori)
            x_list.append(x_sample.permute(0, 2, 1))
            x_ori = x_sample
        return x_list

    def _pre_enc(self, x_list):
        """
        Apply series decomposition if not channel independent.
        Returns two lists: the first (processed input) and the second (residuals).
        """
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x1, x2 = self.preprocess(x)
                out1_list.append(x1)
                out2_list.append(x2)
            return (out1_list, out2_list)


#######################################
#         Forecasting Class
#######################################
class TimeMixerForecast(TimeMixerBase):
    def __init__(self, D1, T1, D2, T2, config_overrides=None):
        # Create default configuration using the modified config.
        config = ConfigForecast(D1, T1, D2, T2)
        if config_overrides:
            # Update config attributes with any overrides.
            for key, value in config_overrides.items():
                setattr(config, key, value)
        self.config = config
        super(TimeMixerForecast, self).__init__(config)

        # Create prediction layers for each scale.
        self.predict_layers = nn.ModuleList()
        # For non-channel-independent branch, create extra residual layers.
        self.out_res_layers = nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        for i in range(self.config.down_sampling_layers + 1):
            seq_len_i = self.seq_len // (self.config.down_sampling_window**i)
            self.predict_layers.append(nn.Linear(seq_len_i, self.config.pred_len))
            if self.channel_independence == 0:
                self.out_res_layers.append(nn.Linear(seq_len_i, seq_len_i))
                self.regression_layers.append(
                    nn.Linear(seq_len_i, self.config.pred_len)
                )

        # Projection layer to map from d_model to forecast channels.
        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(self.config.d_model, 1)
        else:
            # Here we set c_out to D2.
            self.projection_layer = nn.Linear(self.config.d_model, self.config.c_out)
            # Add a channel projection to map from input channels (D1) to forecast channels (D2)
            self.channel_projection = nn.Linear(self.config.enc_in, self.config.c_out)

    def out_projection(self, dec_out, i, out_res):
        """
        For the non-channel independent branch, add a residual correction.
        """
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        # Project residual branch from D1 to c_out (i.e., D2)
        out_res = self.channel_projection(out_res)
        return dec_out + out_res

    def forward(self, x):
        B = x.shape[0]
        # Multi-scale processing.
        x_list = self._multi_scale_process_inputs(x)  # list of [B, T_i, D1]
        x_list, x_res_list = self._pre_enc(x_list)
        enc_out_list = []
        # For each scale, apply normalization and embedding.
        for i, x_scale in enumerate(x_list):
            x_scale = self.normalize_layers[i](x_scale, "norm")
            if self.channel_independence == 1:
                B_temp, T_temp, D_temp = x_scale.shape
                x_scale = (
                    x_scale.permute(0, 2, 1)
                    .contiguous()
                    .reshape(B_temp * D_temp, T_temp, 1)
                )
            enc_out = self.enc_embedding(x_scale, None)
            enc_out_list.append(enc_out)
        # Process through the PastDecomposableMixing blocks.
        for block in self.pdm_blocks:
            enc_out_list = block(enc_out_list)
        # Future multi mixing: for each scale use a predict layer and (if needed) add residual.
        dec_out_list = []
        for i, enc_out in enumerate(enc_out_list):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            if self.channel_independence == 1:
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(
                    B, self.config.c_out, self.config.pred_len
                ).permute(0, 2, 1)
            else:
                dec_out = self.out_projection(dec_out, i, x_res_list[i])
            dec_out_list.append(dec_out)
        # Sum over the scales.
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        # Only apply denormalization if forecast channels match input channels.
        if self.config.enc_in == self.config.c_out:
            dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out


#######################################
#       Classification Class
#######################################
class TimeMixerClassification(TimeMixerBase):
    def __init__(self, D, T, D2, T2, config_overrides=None):
        # Default configuration for classification.
        config = ConfigForecast(
            D, T, D, T2
        )  # D used for both input and (dummy) output here
        config.task_name = "classification"
        config.num_class = 1  # Always output a single number for binary classification.
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)
        self.config = config
        super(TimeMixerClassification, self).__init__(config)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.dropout)
        # Projection: from d_model (after global pooling) to 1.
        self.projection = nn.Linear(config.d_model, config.num_class)

    def forward(self, x):
        # Multi-scale processing
        x_list = self._multi_scale_process_inputs(x)
        enc_out_list = []
        for i, x_scale in enumerate(x_list):
            x_scale = self.normalize_layers[i](x_scale, "norm")
            if self.channel_independence == 1:
                B_temp, T_temp, D_temp = x_scale.shape
                x_scale = (
                    x_scale.permute(0, 2, 1)
                    .contiguous()
                    .reshape(B_temp * D_temp, T_temp, 1)
                )
            enc_out = self.enc_embedding(x_scale, None)
            enc_out_list.append(enc_out)
        for block in self.pdm_blocks:
            enc_out_list = block(enc_out_list)
        # Use the first scale's output.
        enc_out = enc_out_list[0]  # shape: [B, T, d_model]
        output = self.act(enc_out)
        output = self.dropout(output)
        # Global average pooling over the time dimension: [B, d_model]
        output = output.mean(dim=1)
        # Final projection to (B, 1)
        output = self.projection(output)
        return output


class TimeMixerMultiLabelClassification(TimeMixerBase):
    def __init__(self, D, T, num_labels, T2, config_overrides=None, return_logits=False):
        """
        Args:
            D (int):  # input feature dim
            T (int):  # input time steps
            num_labels (int): number of independent labels to predict
            T2 (int): auxiliary horizon/patch length used by TimeMixer configs
            config_overrides (dict|None): optional overrides for the ConfigForecast
            return_logits (bool): if True, forward() returns raw logits
        """
        config = ConfigForecast(D, T, num_labels, T2)
        config.task_name = "multilabel_classification"
        config.num_class = num_labels

        if config_overrides:
            for k, v in config_overrides.items():
                setattr(config, k, v)

        self.config = config
        super().__init__(config)

        self.return_logits = return_logits
        self.act = F.gelu
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.d_model, config.num_class)

    def forward(self, x):
        # Multi-scale processing (same as classification)
        x_list = self._multi_scale_process_inputs(x)
        enc_out_list = []
        for i, x_scale in enumerate(x_list):
            x_scale = self.normalize_layers[i](x_scale, "norm")
            if self.channel_independence == 1:
                B_temp, T_temp, D_temp = x_scale.shape
                x_scale = (
                    x_scale.permute(0, 2, 1)
                    .contiguous()
                    .reshape(B_temp * D_temp, T_temp, 1)
                )
            enc_out = self.enc_embedding(x_scale, None)
            enc_out_list.append(enc_out)

        for block in self.pdm_blocks:
            enc_out_list = block(enc_out_list)

        # Use first scale output: [B, T, d_model]
        enc_out = enc_out_list[0]
        output = self.act(enc_out)
        output = self.dropout(output)
        # Global average pool over time -> [B, d_model]
        output = output.mean(dim=1)
        # Linear head -> [B, num_labels]
        logits = self.projection(output)

        # if self.return_logits:
        #     return logits
        return logits

#######################################
#         Regression Class
#######################################
class TimeMixerRegression(TimeMixerBase):
    def __init__(self, D, T, D2, T2, config_overrides=None):
        # Default configuration for regression.
        config = ConfigForecast(
            D, T, D, T2
        )  # using D for input and as dummy output dimension
        config.task_name = "regression"
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)
        self.config = config
        super(TimeMixerRegression, self).__init__(config)
        # Projection layer: from aggregated d_model to a single regression value.
        self.projection = nn.Linear(config.d_model, 1)

    def forward(self, x):
        x_list = self._multi_scale_process_inputs(x)
        enc_out_list = []
        for i, x_scale in enumerate(x_list):
            x_scale = self.normalize_layers[i](x_scale, "norm")
            if self.channel_independence == 1:
                B_temp, T_temp, D_temp = x_scale.shape
                x_scale = (
                    x_scale.permute(0, 2, 1)
                    .contiguous()
                    .reshape(B_temp * D_temp, T_temp, 1)
                )
            enc_out = self.enc_embedding(x_scale, None)
            enc_out_list.append(enc_out)
        for block in self.pdm_blocks:
            enc_out_list = block(enc_out_list)
        # Use the first scale’s output.
        enc_out = enc_out_list[0]
        # Global average pooling over the time dimension.
        enc_out = enc_out.mean(dim=1)  # [B, d_model]
        output = self.projection(enc_out)  # [B, 1]
        return output


#######################################
#             Main Test
#######################################
if __name__ == "__main__":
    # For forecasting, assume T must be greater than 5.
    # New interface: D1, T1, D2, T2.
    D1, T1, D2, T2 = (
        1000,
        48,
        384,
        24,
    )  # for example, input has 100 features and output 120 features
    batch_size = 4096

    # --- Forecasting Test ---
    x_forecast = torch.randn(batch_size, T1, D1)  # [B, T1, D1]
    forecast_model = TimeMixerForecast(D1, T1, D2, T2)
    forecast_output = forecast_model(x_forecast)
    print("Forecasting output shape:", forecast_output.shape)
    # Expected shape: [batch, T2, D2]

    # --- Classification Test (Binary) ---
    # For classification, we still use D1 and T1 (forecast output dimension is not used).
    x_class = torch.randn(batch_size, T1, D1)
    classification_model = TimeMixerClassification(D1, T1, T2)
    classification_output = classification_model(x_class)
    print("TimeMixerClassification output shape:", classification_output.shape)
    # Expected: (batch_size, 1)

    # --- Regression Test ---
    x_reg = torch.randn(batch_size, T1, D1)
    regression_model = TimeMixerRegression(D1, T1, T2)
    regression_output = regression_model(x_reg)
    print("Regression output shape:", regression_output.shape)
    # Expected shape: [batch, 1]

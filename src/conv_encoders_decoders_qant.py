"""Q.ANT-primitive drop-in replacements for the 1D/2D conv encoders/decoders.

These mirror ``src.conv_encoders_decoders.Conv1DEncoder`` /
``Conv1DDecoder`` / ``Conv2DEncoder`` / ``Conv2DDecoder`` exactly (same
hyperparameters, same architecture, same forward-pass shape logic), but build
every parameterized layer (conv, transpose-conv, batchnorm, maxpool, linear,
relu) from the Q.ANT-dispatched wrappers in :mod:`src.qant_conv_layers` and
:mod:`src.plume_qant_layers`. When every layer's ``qant_backend`` is
``"torch"`` (the default) these are numerically identical to the originals;
setting ``qant_backend="qant"`` (via :func:`src.plume_qant_layers.set_qant_backend`)
routes every op through the CPU-emulated Q.ANT NPU in eval mode.

3D (image-sequence) branches are intentionally not ported: the native Q.ANT
SDK exposes only 2D convolution/pooling primitives, and TokaMark task 3-1
never exercises the 3D branch (see ``src.conv_encoders_decoders.Conv3DEncoder``
for the untouched original, still used verbatim by other tasks).
"""

from __future__ import annotations

import torch.nn as nn

from src.conv_encoders_decoders import (
    compute_compressed_size_encoder,
    compute_list_compressed_size_decoder,
)
from src.plume_qant_layers import QLinear
from src.qant_conv_layers import (
    QBatchNorm1d,
    QBatchNorm2d,
    QConv1d,
    QConv2d,
    QConvTranspose1d,
    QConvTranspose2d,
    QMaxPool1d,
    QMaxPool2d,
    QReLU,
)

PADDING = 1
KERNEL_SIZE = 3
STRIDE = 3
LAYERS_ENCODER = 3
LAYERS_DECODER = 3
BB_FACTOR = 3


# ======================================================================================================================
class QantConv1DEncoder(nn.Module):

    def __init__(self, input_shape, D, layers=LAYERS_ENCODER, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, bb_factor=BB_FACTOR):
        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = input_shape[0]
        self.ts_var = input_shape[1]

        self.ts_comp = compute_compressed_size_encoder(self.ts_var, layers, kernel_size, stride, padding)

        modules = []
        in_channels = self.n_var
        out_channels = D

        modules.append(QBatchNorm1d(in_channels))

        for _ in range(layers):
            modules.append(QConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            modules.append(QReLU())
            modules.append(QMaxPool1d(2, padding=padding))
            modules.append(QBatchNorm1d(out_channels))

            in_channels = out_channels
            out_channels *= 2

        self.cnn = nn.Sequential(*modules)

        final_channels = D * (2 ** (layers - 1))
        self.fc = QLinear(final_channels * self.ts_comp, D * bb_factor)

    def forward(self, x):
        for layer in self.cnn:
            x = layer(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


# ======================================================================================================================
class QantConv1DDecoder(nn.Module):

    def __init__(self, output_shape, D, layers=LAYERS_DECODER, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, bb_factor=BB_FACTOR, output_padding=1):
        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = output_shape[0]
        self.ts_var = output_shape[1]

        self.list_ts_comp = compute_list_compressed_size_decoder(self.ts_var, layers, kernel_size, stride, padding, output_padding)

        final_channels = D * (2 ** (layers - 1))
        self.fc = QLinear(D * bb_factor, final_channels * self.list_ts_comp[0])

        modules = []
        in_channels = final_channels

        for i in range(layers):
            out_channels = in_channels // 2 if i < layers - 1 else self.n_var
            modules.append(
                QConvTranspose1d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, output_padding=output_padding,
                )
            )
            if i < layers - 1:
                modules.append(QReLU())
                modules.append(QBatchNorm1d(out_channels))
            in_channels = out_channels

        self.transposecnn = nn.Sequential(*modules)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.D * (2 ** (self.layers - 1)), self.list_ts_comp[0])

        conv_id = 0
        for layer in self.transposecnn:
            x = layer(x)
            if isinstance(layer, QConvTranspose1d):
                conv_id += 1
                target_ts = self.list_ts_comp[conv_id]
                x = x[:, :, :target_ts]

        return x


# ======================================================================================================================
class QantConv2DEncoder(nn.Module):

    def __init__(self, input_shape, D, layers=LAYERS_ENCODER, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, bb_factor=BB_FACTOR):
        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = input_shape[0]
        self.ts_var = input_shape[1]
        self.height_var = input_shape[2]

        self.ts_comp = compute_compressed_size_encoder(self.ts_var, layers, kernel_size, stride, padding)
        self.height_comp = compute_compressed_size_encoder(self.height_var, layers, kernel_size, stride, padding)

        modules = []
        in_channels = self.n_var
        out_channels = D

        modules.append(QBatchNorm2d(in_channels))

        for _ in range(layers):
            modules.append(QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            modules.append(QReLU())
            modules.append(QMaxPool2d(2, padding=padding))
            modules.append(QBatchNorm2d(out_channels))

            in_channels = out_channels
            out_channels *= 2

        self.cnn = nn.Sequential(*modules)

        final_channels = D * (2 ** (layers - 1))
        self.fc = QLinear(final_channels * self.ts_comp * self.height_comp, D * bb_factor)

    def forward(self, x):
        for layer in self.cnn:
            x = layer(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


# ======================================================================================================================
class QantConv2DDecoder(nn.Module):

    def __init__(self, output_shape, D, layers=LAYERS_DECODER, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, bb_factor=BB_FACTOR, output_padding=1):
        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = output_shape[0]
        self.ts_var = output_shape[1]
        self.height_var = output_shape[2]

        self.list_ts_comp = compute_list_compressed_size_decoder(self.ts_var, layers, kernel_size, stride, padding, output_padding)
        self.list_height_comp = compute_list_compressed_size_decoder(self.height_var, layers, kernel_size, stride, padding, output_padding)

        final_channels = D * (2 ** (layers - 1))
        self.fc = QLinear(D * bb_factor, final_channels * self.list_ts_comp[0] * self.list_height_comp[0])

        modules = []
        in_channels = final_channels

        for i in range(layers):
            out_channels = in_channels // 2 if i < layers - 1 else self.n_var
            modules.append(
                QConvTranspose2d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, output_padding=output_padding,
                )
            )
            if i < layers - 1:
                modules.append(QReLU())
                modules.append(QBatchNorm2d(out_channels))
            in_channels = out_channels

        self.transposecnn = nn.Sequential(*modules)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.D * (2 ** (self.layers - 1)), self.list_ts_comp[0], self.list_height_comp[0])

        conv_id = 0
        for layer in self.transposecnn:
            x = layer(x)
            if isinstance(layer, QConvTranspose2d):
                conv_id += 1
                target_ts = self.list_ts_comp[conv_id]
                target_height = self.list_height_comp[conv_id]
                x = x[:, :, :target_ts, :target_height]

        return x

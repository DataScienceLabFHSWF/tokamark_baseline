"""Q.ANT-primitive version of the CNN baseline (``MultiConv_MLP``).

Architecturally identical to :mod:`src.multi_conv_mlp_model` (same encoder/
decoder branch selection, same MLP backbone, same checkpointed forward
logic), but every conv/pool/batchnorm/linear/relu layer is built from the
Q.ANT-dispatched wrappers in :mod:`src.qant_conv_layers` /
:mod:`src.plume_qant_layers`. With every layer's backend left at the default
``"torch"`` this model is numerically equivalent to ``MultiConv_MLP``; call
``model.set_qant_backend("qant")`` (only valid in eval mode, on CPU, with the
Q.ANT SDK installed) to route inference through the CPU-emulated NPU, exactly
like the ``plume_*`` variants. This provides the first genuinely
apples-to-apples (identical bfloat16 dispatch path) baseline for the Q.ANT
quantization-error study.

3D branches fall back to the original (torch-only) ``Conv3DEncoder`` /
``Conv3DDecoder``, since TokaMark task 3-1 never exercises them and the
native Q.ANT SDK has no 3D convolution primitive.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchinfo import summary

from src.model_transform import _make_dummy_outputs
from src.conv_encoders_decoders import Conv3DEncoder, Conv3DDecoder
from src.conv_encoders_decoders_qant import (
    QantConv1DEncoder,
    QantConv1DDecoder,
    QantConv2DEncoder,
    QantConv2DDecoder,
)
from src.plume_qant_layers import QLinear, set_qant_backend
from src.qant_conv_layers import QReLU
from tokamark.tools.utils import get_device

device = get_device()

padding = 1
kernel_size = 3
stride = 3
layers_encoder = 3
layers_decoder = 3
bb_factor = 2


# ----------------------------------------------------------------------------------------------------------------------
def create_cnn_qant_architecture(dataloader_, dict_metadata, D=16, verbose=True):

    if verbose:
        print("\n\n----------Q.ANT CNN MODEL INITIALIZATION----------\n")

    input_shapes = []
    exogenous_shapes = []
    output_shapes = []

    for l, first_window in enumerate(dataloader_.dataset):
        try:
            input_shapes = [arr.shape for arr in first_window["input"]]
            exogenous_shapes = [arr.shape for arr in first_window["exogenous"]]
            output_shapes = [arr.shape for arr in first_window["y"]]

            if verbose:
                print(f"Shot {first_window['shot_id']} used as reference")
                print(f"Input shapes are: {input_shapes}")
                print(f"Actuator future shapes are: {exogenous_shapes}")
                print(f"Output shapes are: {output_shapes}")

            break
        except Exception as e:
            print(f"Skipping sample {l} because not trainable: {e}")
            continue

    model = MultiConv_MLP_QANT(
        input_shapes=input_shapes,
        exogenous_shapes=exogenous_shapes,
        output_shapes=output_shapes,
        dict_metadata=dict_metadata,
        D=D,
    ).to(device)

    input_sizes = [(2,) + shape for shape in (input_shapes + exogenous_shapes)]

    if verbose:
        summary(model, input_size=input_sizes)

    return model


# ======================================================================================================================
class MultiConv_MLP_QANT(nn.Module):

    def __init__(self, input_shapes, exogenous_shapes, output_shapes, dict_metadata, D=16):
        super().__init__()

        self.D = D
        self.W_in = input_shapes[0][0]

        self.output_shapes = output_shapes
        y = _make_dummy_outputs(output_shapes, dict_metadata)
        output_latent_shapes = [arr.shape for arr in y]
        self.W_out = output_latent_shapes[0][0]

        self.input_branches = nn.ModuleList()
        for var_shape in input_shapes:
            self.input_branches.append(self._make_encoder_branch(var_shape))

        self.encoder_mlp = nn.Sequential(
            QLinear(self.D * bb_factor * len(self.input_branches) * self.W_in, 2 * self.D * bb_factor),
            QReLU(),
            QLinear(2 * self.D * bb_factor, self.D * bb_factor),
        )

        self.decoder_mlp = nn.Sequential(
            QLinear(self.D * bb_factor * (1 + len(exogenous_shapes) * self.W_out), 2 * self.D * bb_factor),
            QReLU(),
            QLinear(2 * self.D * bb_factor, self.D * bb_factor),
        )

        self.exogenous_branches = nn.ModuleList()
        for var_shape in exogenous_shapes:
            self.exogenous_branches.append(self._make_encoder_branch(var_shape))

        self.output_branches = nn.ModuleList()
        for var_shape in output_latent_shapes:
            self.output_branches.append(self._make_decoder_branch(var_shape))

    def _make_encoder_branch(self, var_shape):
        if len(var_shape) == 5:
            return Conv3DEncoder(var_shape[1:], self.D, layers_encoder, kernel_size, stride, padding, bb_factor)
        if len(var_shape) == 4:
            return QantConv2DEncoder(var_shape[1:], self.D, layers_encoder, kernel_size, stride, padding, bb_factor)
        if len(var_shape) == 3:
            return QantConv1DEncoder(var_shape[1:], self.D, layers_encoder, kernel_size, stride, padding, bb_factor)
        raise ValueError(f"Unsupported input shape: {var_shape[1:]}")

    def _make_decoder_branch(self, var_shape):
        if len(var_shape) == 5:
            return Conv3DDecoder(var_shape[1:], self.D, layers_decoder, kernel_size, stride, padding, bb_factor)
        if len(var_shape) == 4:
            return QantConv2DDecoder(var_shape[1:], self.D, layers_decoder, kernel_size, stride, padding, bb_factor)
        if len(var_shape) == 3:
            return QantConv1DDecoder(var_shape[1:], self.D, layers_decoder, kernel_size, stride, padding, bb_factor)
        raise ValueError(f"Unsupported input shape: {var_shape[1:]}")

    def set_qant_backend(self, backend: str) -> None:
        """Recursively set the Q.ANT execution backend on every dispatched layer."""
        set_qant_backend(self, backend)

    @staticmethod
    def _run_cnn_encoder(branch, x):
        B, W = x.shape[:2]
        x = x.view(B * W, *x.shape[2:])
        out = branch(x)
        out = out.view(B, W, -1)
        return out

    @staticmethod
    def _run_cnn_decoder(branch, goal_shape, x):
        B, W = x.shape[:2]
        x = x.reshape(B * W, *x.shape[2:])
        out = branch(x)
        out = out.reshape(B, W * out.shape[2], *out.shape[3:])
        target_len = goal_shape[0]
        out = out[:, -target_len:]
        return out

    def forward(self, *args):
        n_in = len(self.input_branches)
        n_exo = len(self.exogenous_branches)

        inputs = args[:n_in]
        exogenous = args[n_in:n_in + n_exo]

        input_branch_outputs = []
        for branch, x in zip(self.input_branches, inputs):
            out = checkpoint(self._run_cnn_encoder, branch, x, use_reentrant=False)
            input_branch_outputs.append(out)
        input_seq = torch.cat(input_branch_outputs, dim=2)

        exo_branch_outputs = []
        for branch, x in zip(self.exogenous_branches, exogenous):
            out = checkpoint(self._run_cnn_encoder, branch, x, use_reentrant=False)
            exo_branch_outputs.append(out)
        exo_seq = torch.cat(exo_branch_outputs, dim=2) if exo_branch_outputs else None

        B, W_in, D_in = input_seq.shape
        enc_in = input_seq.reshape(B, W_in * D_in)
        context = self.encoder_mlp(enc_in)

        if exo_seq is not None:
            B, W_out, D_exo = exo_seq.shape
            exo = exo_seq.reshape(B, W_out * D_exo)
            decoder_in = torch.cat([context, exo], dim=1)
        else:
            decoder_in = context

        dec_flat = self.decoder_mlp(decoder_in)
        dec_out = dec_flat.unsqueeze(1).repeat(1, self.W_out, 1)

        outputs = []
        for branch, goal_shape in zip(self.output_branches, self.output_shapes):
            out = checkpoint(self._run_cnn_decoder, branch, goal_shape, dec_out, use_reentrant=False)
            outputs.append(out)

        return outputs

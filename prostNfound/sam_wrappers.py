"""
Implements wrappers and registry for Segment Anything Model (SAM) models.
"""

import os

import torch
from torch import nn

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "vendor"))

from .segment_anything.modeling.image_encoder import (
    Attention,
    Block,
    ImageEncoderViT,
    MLPBlock,
)
from .segment_anything.build_sam import sam_model_registry as medsam_model_registry
from .segment_anything.build_sam import sam_model_registry
from .externals.sam_med2d.segment_anything import (
    sam_model_registry as sammed_model_registry,
)
from argparse import Namespace


CHECKPOINT_DIR = os.environ.get(
    "PROSTNFOUND_CHECKPOINT_DIR"
)  # top level checkpoint directory
if CHECKPOINT_DIR is None:
    raise ValueError(
        """Environment variable PROSTNFOUND_CHECKPOINT_DIR must be set. It should be a directory with sam and medsam checkpoints."""
    )


def build_sam():
    """Builds the sam-vit-b model."""
    checkpoint = os.path.join(CHECKPOINT_DIR, "sam_vit_b_01ec64.pth")
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    wrap_with_interpolated_pos_embedding_(model)
    return model


def build_medsam():
    """
    Builds the MedSAM model by building the SAM model and loading the medsam checkpoint.
    """
    checkpoint = os.path.join(CHECKPOINT_DIR, "medsam_vit_b_cpu.pth")
    model = medsam_model_registry["vit_b"](checkpoint=checkpoint)
    type(model.image_encoder).forward = forward_return_features
    wrap_with_interpolated_pos_embedding_(model)
    return model


def build_sammed_2d():
    args = Namespace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.image_size = 256
    args.encoder_adapter = True
    args.sam_checkpoint = os.path.join(CHECKPOINT_DIR, "sam-med2d_b.pth")
    model = sammed_model_registry["vit_b"](args).to(device)
    wrap_with_interpolated_pos_embedding_(model)
    return model


def build_adapter_medsam_256():
    checkpoint = os.path.join(CHECKPOINT_DIR, "medsam_vit_b_cpu.pth")
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)

    model.image_encoder = wrap_image_encoder_with_adapter(
        model.image_encoder, adapter_dim=256
    )
    freeze_non_adapter_layers(model.image_encoder)
    wrap_with_interpolated_pos_embedding_(model)
    return model


def build_adapter_sammed_2d():
    model = build_sammed_2d()
    freeze_non_adapter_layers(model.image_encoder)
    wrap_with_interpolated_pos_embedding_(model)
    return model


def build_adapter_sam():
    model = build_sam()
    model.image_encoder = wrap_image_encoder_with_adapter(
        model.image_encoder, adapter_dim=256
    )
    freeze_non_adapter_layers(model.image_encoder)
    wrap_with_interpolated_pos_embedding_(model)
    return model


class Adapter(nn.Module):
    def __init__(self, feature_dim, adapter_dim, init_scale=1e-3):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(feature_dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, feature_dim)
        self.act = nn.GELU()

        # initializations to make it close to identity function
        nn.init.uniform_(self.down_project.weight, -init_scale, init_scale)
        nn.init.uniform_(self.up_project.weight, -init_scale, init_scale)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        return self.up_project(self.act(self.down_project(x))) + x


class AdapterAttn(nn.Module):
    def __init__(self, attn: Attention, adapter_dim: int, init_scale: float = 1e-3):
        super().__init__()
        self.attn = attn
        embedding_dim = attn.proj.in_features

        self.adapter = Adapter(embedding_dim, adapter_dim, init_scale=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.adapter(x)
        return x


class AdapterMLPBlock(nn.Module):
    def __init__(self, mlp: MLPBlock, adapter_dim: int, init_scale: float = 1e-3):
        super().__init__()

        self.mlp = mlp
        embedding_dim = mlp.lin1.in_features

        self.adapter = Adapter(embedding_dim, adapter_dim, init_scale=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = self.adapter(x)
        return x


def wrap_block_with_adapter(block: Block, adapter_dim: int, init_scale: float = 1e-3):
    block.attn = AdapterAttn(block.attn, adapter_dim, init_scale=init_scale)
    block.mlp = AdapterMLPBlock(block.mlp, adapter_dim, init_scale=init_scale)
    return block


def wrap_image_encoder_with_adapter(
    image_encoder: ImageEncoderViT, adapter_dim: int, init_scale: float = 1e-3
):
    new_blocks = torch.nn.ModuleList()
    for block in image_encoder.blocks:
        new_block = wrap_block_with_adapter(block, adapter_dim, init_scale=init_scale)
        new_blocks.append(new_block)

    image_encoder.blocks = new_blocks

    return image_encoder


def freeze_non_adapter_layers(model: nn.Module):
    for name, param in model.named_parameters():
        if "adapter" not in name.lower():
            param.requires_grad = False

    return model


# class ImageEncoderViTWithInterpolatedPositionalEmbeddingsWrapper(nn.Module):
#     def __init__(self, image_encoder: ImageEncoderViT):
#         super().__init__()
#         self.image_encoder = image_encoder
#         self.neck = image_encoder.neck
#         self.patch_embed = image_encoder.patch_embed
#         self.pos_embed = image_encoder.pos_embed
#
#     def forward(self, x):
#         x = self.image_encoder.patch_embed(x)
#         x = x + self.interpolate_pos_encoding(x)
#         for blk in self.image_encoder.blocks:
#             x = blk(x)
#         x = self.image_encoder.neck(x.permute(0, 3, 1, 2))
#         return x
#
#     def interpolate_pos_encoding(self, x):
#         npatch_in_h = x.shape[1]
#         npatch_in_w = x.shape[2]
#
#         patch_pos_embed = self.image_encoder.pos_embed
#
#         npatch_native_h = patch_pos_embed.shape[1]
#         npatch_native_w = patch_pos_embed.shape[2]
#
#         if npatch_native_h == npatch_in_h and npatch_native_w == npatch_in_w:
#             return self.image_encoder.pos_embed
#
#         w0 = npatch_in_w
#         h0 = npatch_in_h
#         # we add a small number to avoid floating point error in the interpolation
#         # see discussion at https://github.com/facebookresearch/dino/issues/8
#         w0, h0 = w0 + 0.1, h0 + 0.1
#         patch_pos_embed = nn.functional.interpolate(
#             patch_pos_embed.permute(0, 3, 1, 2),
#             scale_factor=(h0 / npatch_native_h, w0 / npatch_native_w),
#             mode='bicubic',
#         )
#         assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
#         patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
#         return patch_pos_embed


def interpolate_pos_encoding(x, pos_embed):
    npatch_in_h = x.shape[1]
    npatch_in_w = x.shape[2]

    patch_pos_embed = pos_embed

    npatch_native_h = patch_pos_embed.shape[1]
    npatch_native_w = patch_pos_embed.shape[2]

    if npatch_native_h == npatch_in_h and npatch_native_w == npatch_in_w:
        return pos_embed

    w0 = npatch_in_w
    h0 = npatch_in_h
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.permute(0, 3, 1, 2),
        scale_factor=(h0 / npatch_native_h, w0 / npatch_native_w),
        mode="bicubic",
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
    return patch_pos_embed


def forward_return_features(image_encoder: ImageEncoderViT, x, return_hiddens=False):
    # "Return hiddens" feature added

    x = image_encoder.patch_embed(x)
    if image_encoder.pos_embed is not None:
        x = x + interpolate_pos_encoding(x, image_encoder.pos_embed)

    hiddens = []
    for blk in image_encoder.blocks:
        x = blk(x)
        if return_hiddens:
            hiddens.append(x)

    x = image_encoder.neck(x.permute(0, 3, 1, 2))

    return (x, hiddens) if return_hiddens else x


def wrap_with_interpolated_pos_embedding_(sam_model):
    type(sam_model.image_encoder).forward = forward_return_features

    # sam_model.image_encoder = ImageEncoderViTWithInterpolatedPositionalEmbeddingsWrapper(sam_model.image_encoder)

from dataclasses import dataclass, field
import typing as tp
import torch
from torch import nn
import logging
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .sam_wrappers import build_medsam, build_sam, build_sammed_2d, build_adapter_medsam_256, build_adapter_sam, build_adapter_sammed_2d
from .utils import PatchView
from itertools import chain
import logging
from timm.models.resnet import resnet10t


class ProstNFound(nn.Module):
    BACKBONE_OPTIONS = [
        "sam",
        "medsam",
        "sam_med2d",
        "adapter_medsam",
        "adapter_sam",
        "adapter_sammed_2d",
    ]

    def __init__(
        self,
        floating_point_prompts: list[str] = [],
        discrete_prompts: list[str] = [],
        discrete_prompt_nvals: list[int] = [],
        use_sparse_cnn_patch_features: bool = False,
        use_sparse_cnn_patch_features_rf: bool = False,
        use_prostate_mask_prompt: bool = False,
        num_data_independent_prompts: int = 0,
        prompt_dropout: float = 0.0,  # dropout for prompt embeddings
        sam_backbone: tp.Literal[
            "sam", "medsam", "sam_med2d", "adapter_medsam"
        ] = "medsam",
        replace_patch_embed: bool = False,
        sparse_cnn_backbone_path: str = None,
        freeze_mask_decoder: bool = False,
        freeze_image_encoder: bool = False,
        freeze_cnn: bool = False,
        img_emb_dropout: float = 0.0,
        cnn_patches_whole_prostate: bool = False,
        pos_embed_cnn_patch: bool = True,
        pool_patch_features: bool = None,
        prompt_embedding_dim=256,
    ):
        super().__init__()
        self.floating_point_prompts = floating_point_prompts
        self.discrete_prompts = discrete_prompts
        self.discrete_prompt_nvals = discrete_prompt_nvals
        self.use_sparse_cnn_patch_features = use_sparse_cnn_patch_features
        self.use_sparse_cnn_patch_features_rf = use_sparse_cnn_patch_features_rf
        self.num_data_independent_prompts = num_data_independent_prompts
        self.use_prostate_mask_prompt = use_prostate_mask_prompt

        if use_sparse_cnn_patch_features and use_sparse_cnn_patch_features_rf:
            raise ValueError(
                "Both sparse_cnn_patch_features and sparse_cnn_patch_features_rf cannot be True"
            )

        self.prompt_dropout = prompt_dropout
        self.replace_patch_embed = replace_patch_embed
        self.cnn_patches_whole_prostate = cnn_patches_whole_prostate
        self.pos_embed_cnn_patch = pos_embed_cnn_patch
        self.pool_patch_features = pool_patch_features
        if replace_patch_embed and sam_backbone != "sam_med2d":
            raise ValueError(
                "replace_patch_embed is only supported for sam_med2d backbone"
            )

        self.sparse_cnn_backbone_path = sparse_cnn_backbone_path

        # for p in prompts:
        #     if not p in self.PROMPT_OPTIONS:
        #         raise ValueError(
        #             f"Unknown prompt option: {p}. Options are {self.PROMPT_OPTIONS}"
        #         )

        # BUILD BACKBONE
        if sam_backbone == "medsam":
            self.medsam_model = build_medsam()
            self.image_size_for_features = 1024
        elif sam_backbone == "adapter_medsam":
            self.medsam_model = build_adapter_medsam_256()
            self.image_size_for_features = 1024
        elif sam_backbone == "sam":
            self.medsam_model = build_sam()
            self.image_size_for_features = 1024
        elif sam_backbone == "adapter_sam":
            self.medsam_model = build_adapter_sam()
            self.image_size_for_features = 1024
        elif sam_backbone == "sam_med2d":
            self.medsam_model = build_sammed_2d()

            if replace_patch_embed:
                self.image_size_for_features = 1024
                # sammed_2d has a different input size. Let's hack the model to accept 1024x1024 images

                new_patch_embed = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.GroupNorm(32, 64),
                    nn.GELU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(32, 64),
                    nn.GELU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(32, 64),
                    nn.GELU(),
                    nn.Conv2d(64, 768, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(32, 768),
                    nn.GELU(),
                    nn.MaxPool2d(4, 4),
                    Rearrange("b c h w -> b h w c"),
                )
                self.medsam_model.image_encoder.patch_embed = new_patch_embed
            else:
                # use the default patch embed which is designed for 256x256 images
                self.image_size_for_features = 256
        elif sam_backbone == "adapter_sammed_2d":
            self.medsam_model = build_adapter_sammed_2d()
            self.image_size_for_features = 256

        self.img_emb_dropout = nn.Dropout(img_emb_dropout)

        if freeze_image_encoder:
            logging.debug("Freezing image encoder")
            for param in self.medsam_model.image_encoder.parameters():
                param.requires_grad = False

        if freeze_mask_decoder:
            logging.debug("Freezing mask decoder")
            for param in self.medsam_model.mask_decoder.parameters():
                param.requires_grad = False

        # ====================================================
        # BUILD PROMPT MODULES
        # ==================================================

        # null prompt - used for prompt dropout
        self.null_prompt = nn.Parameter(torch.zeros(1, prompt_embedding_dim))

        # floating point prompts
        self.floating_point_prompt_modules = torch.nn.ModuleDict()
        for prompt in self.floating_point_prompts:
            self.floating_point_prompt_modules[prompt] = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, prompt_embedding_dim),
            )

        # integer prompts
        self.integer_prompt_modules = torch.nn.ModuleDict()
        for prompt, num_categories in zip(
            self.integer_prompt_modules, self.discrete_prompt_nvals
        ):
            self.integer_prompt_modules[prompt] = nn.Embedding(
                num_categories, prompt_embedding_dim
            )

        # data independent prompts
        if self.num_data_independent_prompts > 0:
            self.data_independent_prompts = nn.Parameter(
                torch.randn(1, num_data_independent_prompts, prompt_embedding_dim)
            )

        # CNN for extracting patch features
        if self.use_sparse_cnn_patch_features or self.use_sparse_cnn_patch_features_rf:
            model = resnet10t(
                in_chans=3,
            )
            model.fc = nn.Identity()
            model = nn.Sequential(nn.InstanceNorm2d(3), model)
            if sparse_cnn_backbone_path is not None:
                state = torch.load(sparse_cnn_backbone_path, map_location="cpu")
                model.load_state_dict(
                    {
                        k.replace("backbone.", ""): v
                        for k, v in state.items()
                        if "backbone" in k
                    }
                )
            self.patch_feature_cnn = model
            if freeze_cnn:
                for param in self.patch_feature_cnn.parameters():
                    param.requires_grad = False

            # project the CNN features to the prompt space
            # self.patch_feature_prompt_module = nn.Linear(512, EMBEDDING_DIM)
            self.patch_feature_prompt_module = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, prompt_embedding_dim),
            )
            self.pad_token = nn.Parameter(
                torch.zeros(prompt_embedding_dim)
            )  # for padding the number of patches to a fixed number in a minibatch

    def forward(
        self,
        image=None,
        rf_image=None,
        prostate_mask=None,
        needle_mask=None,
        return_prompt_embeddings=False,
        **prompts,
    ):
        DEVICE = image.device
        B, C, H, W = image.shape

        if H != self.image_size_for_features or W != self.image_size_for_features:
            image_resized_for_features = torch.nn.functional.interpolate(
                image, size=(self.image_size_for_features, self.image_size_for_features)
            )
        else:
            image_resized_for_features = image

        image_feats = self.medsam_model.image_encoder(image_resized_for_features)
        image_feats = self.img_emb_dropout(image_feats)

        if self.use_prostate_mask_prompt:
            if (
                prostate_mask is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                mask = None
            else:
                B, C, H, W = prostate_mask.shape
                if H != 256 or W != 256:
                    prostate_mask = torch.nn.functional.interpolate(
                        prostate_mask, size=(256, 256)
                    )
                mask = prostate_mask
        else:
            mask = None

        sparse_embedding, dense_embedding = self.medsam_model.prompt_encoder.forward(
            None, None, mask
        )

        # if "dense_cnn_features" in self.prompts:
        #     dense_features = self.patch_feature_cnn[0](image)
        #     dense_features = self.patch_feature_cnn[1].forward_features(dense_features)
        #     dense_features = self.dense_feature_projection(dense_features)
        #     dense_features = torch.nn.functional.interpolate(
        #         dense_features, size=dense_embedding.shape[-2:]
        #     )
        #     if self.training:
        #         dense_features = torch.nn.functional.dropout(dense_features, p=0.5, training=True)
        #     dense_embedding = dense_embedding + dense_features

        sparse_embedding = sparse_embedding.repeat_interleave(len(image), 0)

        for prompt_name, prompt_value in prompts.items():
            if (
                prompt_value is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                # skip this prompt and use the 'null embedding' instead
                prompt_embedding = self.null_prompt.repeat_interleave(len(image), 0)
            else:
                if prompt_name in self.floating_point_prompts:
                    prompt_embedding = self.floating_point_prompt_modules[prompt_name](
                        prompt_value
                    )
                elif prompt_name in self.discrete_prompts:
                    prompt_embedding = self.integer_prompt_modules[prompt_name](
                        prompt_value
                    )
                else:
                    raise ValueError(f"Unknown prompt: {prompt_name}")

            prompt_embedding = prompt_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, prompt_embedding], dim=1)

        if self.num_data_independent_prompts > 0:
            sparse_embedding = torch.cat(
                [
                    sparse_embedding,
                    self.data_independent_prompts.repeat_interleave(B, 0),
                ],
                dim=1,
            )

        if self.use_sparse_cnn_patch_features:
            patch_cnn_sparse_embeddings = self.get_cnn_patch_embedding_bmode(
                image, needle_mask, prostate_mask
            )
            if patch_cnn_sparse_embeddings is not None:
                sparse_embedding = torch.cat(
                    [sparse_embedding, patch_cnn_sparse_embeddings], dim=1
                )

        if self.use_sparse_cnn_patch_features_rf:
            patch_cnn_sparse_embeddings = self.get_cnn_patch_embedding_rf(
                rf_image, needle_mask, prostate_mask
            )
            if patch_cnn_sparse_embeddings is not None:
                sparse_embedding = torch.cat(
                    [sparse_embedding, patch_cnn_sparse_embeddings], dim=1
                )

        mask_logits = self.medsam_model.mask_decoder.forward(
            image_feats,
            self.medsam_model.prompt_encoder.get_dense_pe(),
            sparse_embedding,
            dense_embedding,
            multimask_output=False,
        )[0]

        if return_prompt_embeddings:
            return mask_logits, sparse_embedding, dense_embedding
        else:
            return mask_logits

    def get_cnn_patch_embedding_bmode(self, image, needle_mask, prostate_mask):
        # we need to extract patches from the images.
        DEVICE = image.device
        patches = []
        batch_indices = []
        positions = []
        B = len(image)
        for i in range(B):

            im = image[i].permute(1, 2, 0).cpu().numpy()
            mask = needle_mask[i].permute(1, 2, 0).cpu().numpy()
            prostate_mask_ = prostate_mask[i].permute(1, 2, 0).cpu().numpy()

            if self.cnn_patches_whole_prostate:
                masks = [prostate_mask_]
                thresholds = [0.9]
            else:
                masks = [mask, prostate_mask_]
                thresholds = [0.3, 0.9]

            pv = PatchView.from_sliding_window(
                im,
                window_size=(128, 128),
                stride=(64, 64),
                masks=masks,
                thresholds=thresholds,
            )
            for position, patch in zip(pv.positions, pv):
                patches.append(torch.from_numpy(patch).permute(2, 0, 1))
                positions.append(torch.from_numpy(position))
                batch_indices.append(i)

        patches = torch.stack(patches).to(DEVICE)
        positions = torch.stack(positions).to(DEVICE)
        positions = positions[:, [1, 0]]
        batch_indices = torch.tensor(batch_indices)

        patch_cnn_output = self.patch_feature_cnn(patches)
        patch_cnn_output = self.patch_feature_prompt_module(patch_cnn_output)
        if self.pos_embed_cnn_patch:
            position_encoding_outputs = (
                self.medsam_model.prompt_encoder.pe_layer.forward_with_coords(
                    positions[None, ...], image_size=(1024, 1024)
                )[0]
            )
            patch_cnn_output = patch_cnn_output + position_encoding_outputs

        sparse_embeddings_by_batch = []
        for i in range(B):
            patch_embeddings_for_batch = patch_cnn_output[batch_indices == i]
            if self.pool_patch_features == "mean":
                if len(patch_embeddings_for_batch) == 0:
                    return None
                patch_embeddings_for_batch = torch.mean(
                    patch_embeddings_for_batch, dim=0, keepdim=True
                )
            elif self.pool_patch_features == "max":
                if len(patch_embeddings_for_batch) == 0:
                    return None
                patch_embeddings_for_batch = torch.max(
                    patch_embeddings_for_batch, dim=0, keepdim=True
                ).values
            sparse_embeddings_by_batch.append(patch_embeddings_for_batch)

        max_len = max([len(e) for e in sparse_embeddings_by_batch])
        patch_cnn_sparse_embeddings = torch.zeros(B, max_len, 256, device=DEVICE)
        for i, e in enumerate(sparse_embeddings_by_batch):
            patch_cnn_sparse_embeddings[i, : len(e)] = e
            patch_cnn_sparse_embeddings[i, len(e) :] = self.pad_token[None, None, :]

        if self.prompt_dropout > 0 and self.training:
            for i in range(patch_cnn_sparse_embeddings.shape[1]):
                if torch.rand(1) < self.prompt_dropout:
                    patch_cnn_sparse_embeddings[:, i, :] = (
                        self.null_prompt.repeat_interleave(B, 0)
                    )

        return patch_cnn_sparse_embeddings

    def get_cnn_patch_embedding_rf(self, image, needle_mask, prostate_mask):
        # we need to extract patches from the images.
        DEVICE = image.device
        patches = []
        batch_indices = []
        positions = []

        im_size_mm = 28, 46.06
        B, C, H, W = image.shape
        logging.debug(f"RF shape: {image.shape}")
        im_size_px = H, W
        patch_size_mm = 5, 5
        if not self.cnn_patches_whole_prostate:
            patch_stride_mm = 1, 1
        else:
            patch_stride_mm = 2, 2
        patch_size_px = int(patch_size_mm[0] / im_size_mm[0] * im_size_px[0]), int(
            patch_size_mm[1] / im_size_mm[1] * im_size_px[1]
        )
        patch_stride_px = int(patch_stride_mm[0] / im_size_mm[0] * im_size_px[0]), int(
            patch_stride_mm[1] / im_size_mm[1] * im_size_px[1]
        )
        logging.debug(f"Patch size: {patch_size_px}")

        B = len(image)
        for i in range(B):

            im = image[i].permute(1, 2, 0).cpu().numpy()
            mask = needle_mask[i].permute(1, 2, 0).cpu().numpy()
            prostate_mask_ = prostate_mask[i].permute(1, 2, 0).cpu().numpy()

            if self.cnn_patches_whole_prostate:
                masks = [prostate_mask_]
                thresholds = [0.9]
            else:
                masks = [mask]
                thresholds = [0.6]

            pv = PatchView.from_sliding_window(
                im,
                window_size=patch_size_px,
                stride=patch_stride_px,
                masks=masks,
                thresholds=thresholds,
                align_to="topright",
            )
            for position, patch in zip(pv.positions, pv):
                patches.append(torch.from_numpy(patch).permute(2, 0, 1))
                positions.append(torch.from_numpy(position))
                batch_indices.append(i)

        logging.debug(f"Extracted {len(patches)} patches from {B} rf images")
        if len(patches) == 0:
            return None

        patches = torch.stack(patches).to(self.device)
        # patches should be resized to 256 by 256 as used in the RF CNNs
        patches = torch.nn.functional.interpolate(
            patches, size=(256, 256), mode="bilinear"
        )

        positions = torch.stack(positions).to(DEVICE)
        positions = positions[:, [1, 0]]
        batch_indices = torch.tensor(batch_indices)

        patch_cnn_output = self.patch_feature_cnn(patches)
        patch_cnn_output = self.patch_feature_prompt_module(patch_cnn_output)

        if self.pos_embed_cnn_patch:
            position_encoding_outputs = (
                self.medsam_model.prompt_encoder.pe_layer.forward_with_coords(
                    positions[None, ...], image_size=(1024, 1024)
                )[0]
            )
            patch_cnn_output = patch_cnn_output + position_encoding_outputs

        sparse_embeddings_by_batch = []
        for i in range(B):
            patch_embeddings_for_batch = patch_cnn_output[batch_indices == i]  # N x 256
            if self.pool_patch_features == "mean":
                if len(patch_embeddings_for_batch) == 0:
                    return None  # no patches found
                patch_embeddings_for_batch = torch.mean(
                    patch_embeddings_for_batch, dim=0, keepdim=True
                )
            elif self.pool_patch_features == "max":
                if len(patch_embeddings_for_batch) == 0:
                    return None
                patch_embeddings_for_batch = torch.max(
                    patch_embeddings_for_batch, dim=0, keepdim=True
                ).values
            sparse_embeddings_by_batch.append(patch_embeddings_for_batch)

        max_len = max([len(e) for e in sparse_embeddings_by_batch])
        patch_cnn_sparse_embeddings = torch.zeros(B, max_len, 256, device=DEVICE)
        for i, e in enumerate(sparse_embeddings_by_batch):
            patch_cnn_sparse_embeddings[i, : len(e)] = e
            patch_cnn_sparse_embeddings[i, len(e) :] = self.pad_token[None, None, :]

        if self.prompt_dropout > 0 and self.training:
            for i in range(patch_cnn_sparse_embeddings.shape[1]):
                if torch.rand(1) < self.prompt_dropout:
                    patch_cnn_sparse_embeddings[:, i, :] = (
                        self.null_prompt.repeat_interleave(B, 0)
                    )

        B, N, C = patch_cnn_sparse_embeddings.shape
        if self.pool_patch_features == "transformer":
            patch_cnn_sparse_embeddings = self.patch_aggregator(
                patch_cnn_sparse_embeddings
            )
            patch_cnn_sparse_embeddings = patch_cnn_sparse_embeddings.mean(
                dim=1, keepdim=True
            )

        return patch_cnn_sparse_embeddings

    def train(self, mode: bool = True):
        super().train(mode)
        # always keep cnn in eval mode - otherwise batch norm might interfere.
        if (
            (
                self.use_sparse_cnn_patch_features
                or self.use_sparse_cnn_patch_features_rf
            )
            and self.sparse_cnn_backbone_path is not None
            and self.patch_feature_cnn is not None
        ):
            self.patch_feature_cnn.eval()

    def get_params_groups(self):

        encoder_parameters = [
            p
            for (k, p) in self.medsam_model.image_encoder.named_parameters()
            if "neck" not in k
        ]

        warmup_parameters = []
        # warmup components from backbone
        warmup_parameters = chain(
            warmup_parameters, self.medsam_model.mask_decoder.parameters()
        )
        warmup_parameters = chain(
            warmup_parameters, self.medsam_model.prompt_encoder.parameters()
        )
        warmup_parameters = chain(
            warmup_parameters, self.medsam_model.image_encoder.neck.parameters()
        )
        # null prompt
        warmup_parameters = chain(warmup_parameters, [self.null_prompt])
        # floating point prompts
        for module in self.floating_point_prompt_modules.values():
            warmup_parameters = chain(warmup_parameters, module.parameters())
        # data independent prompts
        for module in self.integer_prompt_modules.values():
            warmup_parameters = chain(warmup_parameters, module.parameters())
        # patch prompts
        if self.use_sparse_cnn_patch_features or self.use_sparse_cnn_patch_features_rf:
            warmup_parameters = chain(
                warmup_parameters, self.patch_feature_prompt_module.parameters()
            )
        # data independent prompts
        if self.num_data_independent_prompts > 0:
            warmup_parameters = chain(
                warmup_parameters, [self.data_independent_prompts]
            )

        cnn_parameters = (
            self.patch_feature_cnn.parameters()
            if self.use_sparse_cnn_patch_features
            or self.use_sparse_cnn_patch_features_rf
            else []
        )

        return encoder_parameters, warmup_parameters, cnn_parameters

    @property
    def device(self):

        return next(self.parameters()).device
    

@dataclass(frozen=True)
class ProstNFoundConfig:
    """ProstNFound model configuration.
    
    Args: 
        backbone: type of sam backbone to use
        floating_point_prompts: List of floating point prompts to use. All of these should be keys in the prompt table
        discrete_prompts: List of discrete prompts to use. All of these should be keys in the prompt table
        discrete_prompts_nvals: For each discrete prompt, the number of values it can take
        use_sparse_cnn_patch_features: If True, uses the sparse CNN patch features as input to the model.
        use_sparse_cnn_patch_features_rf: If True, uses the sparse CNN patch features as input to the model.
        num_data_independent_prompts: The number of data independent prompts to use. Data independent are learnable prompts that don't depend on input (ie. prompt tuning)
        prompt_dropout: The dropout to use for prompt embeddings. If set greater than zero, this will randomly drop each prompt embedding by replacing it with a `null prompt`. 
            Could help to reduce excessive reliance on prompts.
        pool_patch_features: The pooling operation to use for patch features. If None, does no pooling. We recommend max. Only applies if using one of the patch feature modes. 
        sparse_cnn_backbone_path: The path to the sparse CNN backbone weights to use. Only applies if using one of the patch feature modes.
    """
    backbone: tp.Literal[*ProstNFound.BACKBONE_OPTIONS] = "medsam"
    floating_point_prompts: list[str] = field(default_factory=lambda:[])
    discrete_prompts: list[str] = field(default_factory=lambda:[]) 
    discrete_prompts_nvals: list[int] = field(default_factory=lambda:[]) 
    use_sparse_cnn_patch_features: bool = False 
    use_sparse_cnn_patch_features_rf: bool = False 
    num_data_independent_prompts: int = 0 
    prompt_dropout: float = 0.0 
    pool_patch_features: str | None = None #
    sparse_cnn_backbone_path: str | None = None 


def build_prostnfound(cfg: ProstNFoundConfig): 
    logging.info(f'Building ProstNFound model with config: {cfg}')

    model = ProstNFound(
        floating_point_prompts=cfg.floating_point_prompts,
        discrete_prompts=cfg.discrete_prompts,
        discrete_prompt_nvals=cfg.discrete_prompts_nvals,
        use_sparse_cnn_patch_features=cfg.use_sparse_cnn_patch_features,
        use_sparse_cnn_patch_features_rf=cfg.use_sparse_cnn_patch_features_rf,
        num_data_independent_prompts=cfg.num_data_independent_prompts,
        prompt_dropout=cfg.prompt_dropout,
        sam_backbone=cfg.backbone,
        sparse_cnn_backbone_path=cfg.sparse_cnn_backbone_path,
        pool_patch_features=cfg.pool_patch_features,
    )

    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    return model


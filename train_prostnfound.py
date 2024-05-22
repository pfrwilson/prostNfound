import argparse
import logging
import os
import typing as tp
from argparse import ArgumentParser
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import wandb
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from src.data_factory import DataLoaderFactory
from torch.nn import functional as F
from tqdm import tqdm
from rich_argparse import ArgumentDefaultsRichHelpFormatter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.utils import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
    DataFrameCollector, 
    calculate_metrics, 
    PatchView
)
import json
from skimage.transform import resize
from skimage.filters import gaussian
from src.sam_wrappers import (
    build_adapter_medsam_256,
    build_adapter_sam,
    build_adapter_sammed_2d,
    build_medsam,
    build_sam,
    build_sammed_2d,
)
from einops.layers.torch import Rearrange
from timm.models.resnet import resnet10t
from src.transformer import TransformerEncoder
from itertools import chain


def parse_args():
    # fmt: off
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsRichHelpFormatter)

    # data args
    group = parser.add_argument_group("Data - cohort selection", "Arguments related to cohort selection")
    group.add_argument("--fold", type=int, default=None, help="The fold to use. If not specified, uses leave-one-center-out cross-validation.")
    group.add_argument("--n_folds", type=int, default=None, help="The number of folds to use for cross-validation.")
    group.add_argument("--test_center", type=str, default=None, 
                        help="If not None, uses leave-one-center-out cross-validation with the specified center as test. If None, uses k-fold cross-validation.")
    group.add_argument("--val_seed", type=int, default=0, 
                       help="The seed to use for validation split.")            
    group.add_argument("--undersample_benign_ratio", type=lambda x: float(x) if not x.lower() == 'none' else None, default=None,
                       help="""If not None, undersamples benign cores with the specified ratio.""")
    group.add_argument("--min_involvement_train", type=float, default=0.0,
                       help="""The minimum involvement threshold to use for training.""")
    group.add_argument("--remove_benign_cores_from_positive_patients", action="store_true", help="If True, removes benign cores from positive patients (training only).")
    group.add_argument("--limit_train_data", type=float, default=1., 
                       help="""If less than 1, chooses a center-balanced subset of the original train data to train with. The value given is the fraction of the original data to use.""")
    group.add_argument("--train_subsample_seed", type=int, default=42, help="The seed to use for subsampling the training data (if limit_train_data < 1).")
    group.add_argument("--splits_json_file", help="If provided, overrides all of the above argument and looks up the core_ids for train, validation and test from the given splits file. The file should be a json file with keys 'train', 'val' and 'test' each containing a list of core_ids.") 
    
    group = parser.add_argument_group("Data - Processing", "Arguments related to data loading and preprocessing")
    group.add_argument("--batch_size", type=int, default=4, 
        help="The batch size to use for training. Often limited by GPU size - if you want a larger effective batch size you can also adjust `--accumulate_grad_steps`.")
    group.add_argument("--augmentations", type=str, default="translate", help="The augmentations to use for training. We found random translation to boost performance compared to no augmentations.")
    group.add_argument("--image_size", type=int, default=1024, help="The size to use for the images.")
    group.add_argument("--mask_size", type=int, default=256, help="The size to use for the masks.")
    
    group.add_argument("--rf_as_bmode", action="store_true", help="If True, uses the RF images as B-mode images. (Hack to be used to test foundation model performance on RF directly)")
    
    parser.add_argument('--custom_prompt_table_path', type=str, default=None, help="The path to the custom prompt table to use.")

    group = parser.add_argument_group("Training", "Arguments related to training.")
    group.add_argument("--optimizer", type=str, default="adamw", help="The optimizer to use for training.")

    group.add_argument("--cnn_lr", type=float, default=1e-5, help="The learning rate to use for the CNN. (Only used if sparse_cnn_patch_features of sparse_cnn_patch_features_rf is in the list of prompts.)")
    group.add_argument("--cnn_frozen_epochs", type=int, default=0, help="The number of frozen epochs for the cnn.")
    group.add_argument("--cnn_warmup_epochs", type=int, default=5, help="The number of linear warmup epochs for the cnn.")
    group.add_argument("--encoder_lr", type=float, default=1e-5, help="The learning rate to use for the encoder.")
    group.add_argument("--encoder_frozen_epochs", type=int, default=0, help="The number of frozen epochs for the encoder.")
    group.add_argument("--encoder_warmup_epochs", type=int, default=5, help="The number of linear warmup epochs for the encoder.")
    group.add_argument("--main_lr", type=float, default=1e-5, help="The learning rate to use for the main of the model (mask decoder and patch decoder components)")
    group.add_argument("--main_frozen_epochs", type=int, default=0, help="The number of frozen epochs for the main part of the model.")
    group.add_argument("--main_warmup_epochs", type=int, default=5, help="The number of linear warmup epochs for the main part of the model.")

    group.add_argument("--wd", type=float, default=0, 
        help="The weight decay to use for training. We found weight decay can degrade performance (likely due to forgetting foundation model pretraining) so it is off by default.")
    group.add_argument("--epochs", type=int, default=30, help="The number of epochs for the training and learning rate annealing.")
    group.add_argument("--cutoff_epoch", type=int, default=None, help="If not None, the training will stop after this epoch, but this will not affect the learning rate scheduler.")
    group.add_argument("--accumulate_grad_steps", type=int, default=8, 
        help="""The number of gradient accumulation steps to use. Can be used to increase the effective batch size when GPU memory is limited.""")
    group.add_argument("--run_test", default=False, action="store_true", help="If True, runs the test set. Should disable for experiments related to model selection (e.g. hyperparameter tuning)")
    group.add_argument("--test_every_epoch", action="store_true", 
        help="Only used if `--run_test` is set. If this is set, runs the test set every epoch. Otherwise, only runs it when a new best validation score is achieved.")

    # MODEL
    ProstNFound.add_arguments(parser)

    # LOSS
    parser.add_argument("--n_loss_terms", type=int, default=1, help="""The number of loss terms to use.
                        Although we found no benefit beyond using a single masked CE loss, the code supports multiple loss terms.""")
    args, _ = parser.parse_known_args()
    n_loss_terms = args.n_loss_terms
    for i in range(n_loss_terms):
        group = parser.add_argument_group(f"Loss term {i}", f"Arguments related to loss term {i}.")
        group.add_argument(f"--loss_{i}_name", type=str, default="valid_region", choices=('valid_region',), help="The name of the loss function to use."),
        group.add_argument(f"--loss_{i}_base_loss_name", type=str, default="ce", 
                           choices=('ce', 'gce', 'mae', 'mil'), 
                           help="""The name of the lower-level loss function to use. Our experiments showed best performance with simple CE loss, but 
                           due to weak supervision and label noise, we also experimented with GCE, MAE, and MIL. However, we found no benefit beyond using CE loss.""")
        def str2bool(str): 
            return True if str.lower() == 'true' else False
        group.add_argument(f"--loss_{i}_pos_weight", type=float, default=1.0, help="""The positive class weight for the loss function. If using a large ratio of benign to
                           cancer cores in training, it is recommended to increase this value to 2 or 3 to account for the class imbalance.""")
        group.add_argument(f"--loss_{i}_prostate_mask", type=str2bool, default=True, help="If True, the loss will only be applied inside the prostate mask.")
        group.add_argument(f"--loss_{i}_needle_mask", type=str2bool, default=True, help="If True, the loss will only be applied inside the needle mask.")
        group.add_argument(f"--loss_{i}_weight", type=float, default=1.0, help="The weight to use for the loss function.")

    # WANDB
    group = parser.add_argument_group("Wandb", "Arguments related to wandb.")
    group.add_argument("--project", type=str, default="miccai2024", help="The wandb project to use.")
    group.add_argument("--group", type=str, default=None, help="The wandb group to use.")
    group.add_argument("--name", type=str, default=None, help="The wandb name to use.")
    group.add_argument("--log_images", action="store_true", help="If True, logs images to wandb.")
    group.add_argument("--tags", type=str, nargs="+", default=[], help="The tags to use for wandb.")

    # MISC
    group = parser.add_argument_group("Misc", "Miscellaneous arguments.")
    group.add_argument("--encoder_weights_path", type=str, default=None, help="The path to the encoder weights to use. If None, uses the Foundation Model initialization")
    group.add_argument("--encoder_load_mode", type=str, default="none", choices=("dino_medsam", "ibot_medsam", "image_encoder", "none"), help="The mode to use for loading the encoder weights.")
    group.add_argument("--seed", type=int, default=42, help="The seed to use for training.")
    group.add_argument("--use_amp", action="store_true", help="If True, uses automatic mixed precision.")
    group.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="The device to use for training")
    group.add_argument("--exp_dir", type=str, default="experiments/default", help="The directory to use for the experiment.")
    group.add_argument("--checkpoint_dir", type=str, default=None, help="The directory to use for the checkpoints. If None, does not save checkpoints.")
    group.add_argument("--debug", action="store_true", help="If True, runs in debug mode.")
    group.add_argument("--save_weights", choices=("best", "all"), default="best", help="The mode to use for saving weights.")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit')

    args = parser.parse_args()
    return args
    # fmt: on


class Experiment:
    def __init__(self, config):
        self.config = config

    def setup(self):
        logging.basicConfig(
            level=logging.INFO if not self.config.debug else logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logging.info("Setting up experiment")
        os.makedirs(self.config.exp_dir, exist_ok=True)
        logging.info("Running in directory: " + self.config.exp_dir)

        if self.config.debug:
            self.config.name = "debug"
        wandb.init(
            project=self.config.project,
            group=self.config.group,
            name=self.config.name,
            config=self.config,
            tags=self.config.tags,
        )
        logging.info("Wandb initialized")
        logging.info("Wandb url: " + wandb.run.url)

        if self.config.checkpoint_dir is not None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            self.exp_state_path = os.path.join(
                self.config.checkpoint_dir, "experiment_state.pth"
            )
            if os.path.exists(self.exp_state_path):
                logging.info("Loading experiment state from experiment_state.pth")
                self.state = torch.load(self.exp_state_path)
            else:
                logging.info("No experiment state found - starting from scratch")
                self.state = None
        else:
            self.exp_state_path = None
            self.state = None

        set_global_seed(self.config.seed)

        self.setup_data()

        self.setup_model()
        if self.state is not None:
            self.model.load_state_dict(self.state["model"])

        self.setup_optimizer()
        if self.state is not None:
            self.optimizer.load_state_dict(self.state["optimizer"])
            self.lr_scheduler.load_state_dict(self.state["lr_scheduler"])

        self.gradient_scaler = torch.cuda.amp.GradScaler()
        if self.state is not None:
            self.gradient_scaler.load_state_dict(self.state["gradient_scaler"])

        self.epoch = 0 if self.state is None else self.state["epoch"]
        logging.info(f"Starting at epoch {self.epoch}")
        self.best_score = 0 if self.state is None else self.state["best_score"]
        logging.info(f"Best score so far: {self.best_score}")
        if self.state is not None:
            rng_state = self.state["rng"]
            set_all_rng_states(rng_state)

    def setup_model(self):
        logging.info("Setting up model")

        self.model = ProstNFound.from_args(self.config)

        self.model.to(self.config.device)
        torch.compile(self.model)

        logging.info("Model setup complete")
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        logging.info(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        # setup criterion
        loss_terms = []
        loss_weights = []
        for i in range(self.config.n_loss_terms):
            loss_name = getattr(self.config, f"loss_{i}_name")
            base_loss_name = getattr(self.config, f"loss_{i}_base_loss_name")
            loss_pos_weight = getattr(self.config, f"loss_{i}_pos_weight")
            loss_prostate_mask = getattr(self.config, f"loss_{i}_prostate_mask")
            loss_needle_mask = getattr(self.config, f"loss_{i}_needle_mask")
            loss_weight = getattr(self.config, f"loss_{i}_weight")

            if loss_name == "valid_region":
                loss_terms.append(
                    CancerDetectionValidRegionLoss(
                        base_loss=base_loss_name,
                        loss_pos_weight=loss_pos_weight,
                        prostate_mask=loss_prostate_mask,
                        needle_mask=loss_needle_mask,
                    )
                )
                loss_weights.append(loss_weight)
            else:
                raise ValueError(f"Unknown loss name: {loss_name}")

        self.loss_fn = MultiTermCanDetLoss(loss_terms, loss_weights)

    def setup_optimizer(self):
        

        (
            encoder_parameters,
            warmup_parameters,
            cnn_parameters,
        ) = self.model.get_params_groups()

        # total_epochs = self.config.epochs
        # encoder_frozen_epochs = self.config.warmup_epochs
        # warmup_epochs = 5
        # niter_per_ep = len(self.train_loader)
        # warmup_lr_factor = self.config.warmup_lr / self.config.lr
        params = [
            {"params": encoder_parameters, "lr": self.config.encoder_lr},
            {"params": warmup_parameters, "lr": self.config.main_lr},
            {"params": cnn_parameters, "lr": self.config.cnn_lr},
        ]

        class LRCalculator:
            def __init__(self, frozen_epochs, warmup_epochs, total_epochs, niter_per_ep):
                self.frozen_epochs = frozen_epochs
                self.warmup_epochs = warmup_epochs
                self.total_epochs = total_epochs
                self.niter_per_ep = niter_per_ep

            def __call__(self, iter): 
                if iter < self.frozen_epochs * self.niter_per_ep: 
                    return 0
                elif iter < (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep: 
                    return (iter - self.frozen_epochs * self.niter_per_ep) / (self.warmup_epochs * self.niter_per_ep)
                else: 
                    cur_iter = iter - (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
                    total_iter = (self.total_epochs - self.warmup_epochs - self.frozen_epochs) * self.niter_per_ep
                    return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))


        # def compute_lr_multiplier(iter, is_encoder_or_cnn=True):
        #     if iter < encoder_frozen_epochs * niter_per_ep:
        #         if is_encoder_or_cnn:
        #             return 0
        #         else:
        #             if iter < warmup_epochs * niter_per_ep:
        #                 return (iter * warmup_lr_factor) / (
        #                     warmup_epochs * niter_per_ep
        #                 )
        #             else:
        #                 cur_iter_in_frozen_phase = iter - warmup_epochs * niter_per_ep
        #                 total_iter_in_frozen_phase = (
        #                     encoder_frozen_epochs - warmup_epochs
        #                 ) * niter_per_ep
        #                 return (
        #                     0.5
        #                     * (
        #                         1
        #                         + np.cos(
        #                             np.pi
        #                             * cur_iter_in_frozen_phase
        #                             / (total_iter_in_frozen_phase)
        #                         )
        #                     )
        #                     * warmup_lr_factor
        #                 )
        #     else:
        #         iter -= encoder_frozen_epochs * niter_per_ep
        #         if iter < warmup_epochs * niter_per_ep:
        #             return iter / (warmup_epochs * niter_per_ep)
        #         else:
        #             cur_iter = iter - warmup_epochs * niter_per_ep
        #             total_iter = (
        #                 total_epochs - warmup_epochs - encoder_frozen_epochs
        #             ) * niter_per_ep
        #             return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))

        self.optimizer = AdamW(params, weight_decay=self.config.wd)
        

        # self.lr_scheduler = LambdaLR(
        #     self.optimizer,
        #     [
        #         lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=True),
        #         lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=False),
        #         lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=True),
        #     ],
        # )
        self.lr_scheduler = LambdaLR(self.optimizer, 
            [LRCalculator(self.config.encoder_frozen_epochs, self.config.encoder_warmup_epochs, self.config.epochs, len(self.train_loader)),
             LRCalculator(self.config.main_frozen_epochs, self.config.main_warmup_epochs, self.config.epochs, len(self.train_loader)),
             LRCalculator(self.config.cnn_frozen_epochs, self.config.cnn_warmup_epochs, self.config.epochs, len(self.train_loader))
            ])
        
    def setup_data(self):
        logging.info("Setting up data")

        data_factory = DataLoaderFactory(
            fold=self.config.fold,
            n_folds=self.config.n_folds,
            test_center=self.config.test_center,
            undersample_benign_ratio=self.config.undersample_benign_ratio,
            min_involvement_train=self.config.min_involvement_train,
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            mask_size=self.config.mask_size,
            augmentations=self.config.augmentations,
            remove_benign_cores_from_positive_patients=self.config.remove_benign_cores_from_positive_patients,
            val_seed=self.config.val_seed,
            limit_train_data=self.config.limit_train_data
            if self.config.limit_train_data < 1
            else None,
            train_subset_seed=self.config.train_subsample_seed,
            rf_as_bmode=self.config.rf_as_bmode,
            include_rf= "sparse_cnn_patch_features_rf" in self.config.prompts,
            splits_file=self.config.splits_json_file
        )
        self.train_loader = data_factory.train_loader()
        self.val_loader = data_factory.val_loader()
        self.test_loader = data_factory.test_loader()
        logging.info(f"Number of training batches: {len(self.train_loader)}")
        logging.info(f"Number of validation batches: {len(self.val_loader)}")
        logging.info(f"Number of test batches: {len(self.test_loader)}")
        logging.info(f"Number of training samples: {len(self.train_loader.dataset)}")
        logging.info(f"Number of validation samples: {len(self.val_loader.dataset)}")
        logging.info(f"Number of test samples: {len(self.test_loader.dataset)}")

        # dump core_ids to file
        train_core_ids = self.train_loader.dataset.core_ids
        val_core_ids = self.val_loader.dataset.core_ids
        test_core_ids = self.test_loader.dataset.core_ids

        with open(os.path.join(self.config.exp_dir, "train_core_ids.txt"), "w") as f:
            f.write("\n".join(train_core_ids))
        with open(os.path.join(self.config.exp_dir, "val_core_ids.txt"), "w") as f:
            f.write("\n".join(val_core_ids))
        with open(os.path.join(self.config.exp_dir, "test_core_ids.txt"), "w") as f:
            f.write("\n".join(test_core_ids))

        wandb.save(os.path.join(self.config.exp_dir, "train_core_ids.txt"))
        wandb.save(os.path.join(self.config.exp_dir, "val_core_ids.txt"))
        wandb.save(os.path.join(self.config.exp_dir, "test_core_ids.txt"))

        if self.config.custom_prompt_table_path is not None: 
            print("Loading custom prompt table")
            self.custom_prompt_table = pd.read_csv(self.config.custom_prompt_table_path)
            self.custom_prompt_table.set_index('core_id', inplace=True, drop=True)
            columns = self.custom_prompt_table.columns
            self.custom_prompt_table_col = columns[0]
            print(f"Using custom prompt {self.custom_prompt_table_col}")
            self.avg_custom_prompt = self.custom_prompt_table[self.custom_prompt_table_col].mean()

    def get_custom_prompts(self, core_ids): 
        custom_prompts = []
        for core_id in core_ids:
            if core_id not in self.custom_prompt_table.index:
                logging.warning(f"Core id {core_id} not found in custom prompt table")
                custom_prompts.append(self.avg_custom_prompt)
                continue
            custom_prompt = self.custom_prompt_table.loc[core_id, self.custom_prompt_table_col].tolist()
            if isinstance(custom_prompt, list):
                custom_prompt = custom_prompt[0]
            custom_prompts.append(custom_prompt)
        return torch.tensor(custom_prompts, dtype=torch.float, device=self.config.device).unsqueeze(1)

    def run(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.epochs):
            if (
                self.config.cutoff_epoch is not None
                and self.epoch > self.config.cutoff_epoch
            ):
                break
            logging.info(f"Epoch {self.epoch}")
            self.save_experiment_state()

            self.run_train_epoch(self.train_loader, desc="train")

            val_metrics = self.run_eval_epoch(self.val_loader, desc="val")

            if val_metrics is not None:
                tracked_metric = val_metrics["val/auc_high_involvement"]
                new_record = tracked_metric > self.best_score
            else:
                new_record = None

            if new_record:
                self.best_score = tracked_metric
                logging.info(f"New best score: {self.best_score}")

            if self.config.run_test and (new_record or self.config.test_every_epoch):
                self.training = False
                logging.info("Running test set")
                metrics = self.run_eval_epoch(self.test_loader, desc="test")
                test_score = metrics["test/auc_high_involvement"]
            else:
                test_score = None

            self.save_model_weights(score=test_score, is_best_score=new_record)

        logging.info("Finished training")
        self.teardown()

    def run_train_epoch(self, loader, desc="train"):
        # setup epoch
        self.model.train()
        

        accumulator = DataFrameCollector()

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            batch_for_image_generation = (
                batch.copy()
            )  # we pop some keys from batch, so we keep a copy for image generation
            if self.config.debug and train_iter > 10:
                break

            # extracting relevant data from the batch
            bmode = batch.pop("bmode").to(self.config.device)
            needle_mask = batch.pop("needle_mask").to(self.config.device)
            prostate_mask = batch.pop("prostate_mask").to(self.config.device)

            psa = batch["psa"].to(self.config.device)
            age = batch["age"].to(self.config.device)
            label = batch["label"].to(self.config.device)
            involvement = batch["involvement"].to(self.config.device)
            family_history = batch["family_history"].to(self.config.device)
            anatomical_location = batch["loc"].to(self.config.device)
            approx_psa_density = batch["approx_psa_density"].to(self.config.device)

            core_ids = batch["core_id"]
            if self.config.custom_prompt_table_path is not None:
                custom_prompts = self.get_custom_prompts(core_ids)
            else: 
                custom_prompts = None

            if 'rf' in batch: 
                rf = batch.pop("rf").to(self.config.device)
            else: 
                rf = None 

            B = len(bmode)
            task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

            # run the model
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # forward pass
                heatmap_logits = self.model(
                    bmode,
                    task_id=task_id,
                    anatomical_location=anatomical_location,
                    psa=psa,
                    age=age,
                    family_history=family_history,
                    prostate_mask=prostate_mask,
                    needle_mask=needle_mask,
                    approx_psa_density=approx_psa_density,
                    rf_image = rf, 
                    custom=custom_prompts,
                )

                if torch.any(torch.isnan(heatmap_logits)):
                    logging.warning("NaNs in heatmap logits")
                    breakpoint()

                # loss calculation
                loss = self.loss_fn(
                    heatmap_logits, prostate_mask, needle_mask, label, involvement
                )

                # compute predictions
                masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
                predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)
                mean_predictions_in_needle = []
                for j in range(B):
                    mean_predictions_in_needle.append(
                        predictions[batch_idx == j].sigmoid().mean()
                    )
                mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)

                prostate_masks = prostate_mask > 0.5
                predictions, batch_idx = MaskedPredictionModule()(
                    heatmap_logits, prostate_masks
                )
                mean_predictions_in_prostate = []
                for j in range(B):
                    mean_predictions_in_prostate.append(
                        predictions[batch_idx == j].sigmoid().mean()
                    )
                mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate)

            loss = loss / self.config.accumulate_grad_steps
            # backward pass
            if self.config.use_amp:
                logging.debug("Backward pass")
                self.gradient_scaler.scale(loss).backward()
            else:
                logging.debug("Backward pass")
                loss.backward()

            # gradient accumulation and optimizer step
            if self.config.debug:
                for param in self.optimizer.param_groups[1]["params"]:
                    break
                logging.debug(param.data.view(-1)[0])

            if (train_iter + 1) % self.config.accumulate_grad_steps == 0:
                logging.debug("Optimizer step")
                if self.config.use_amp:
                    self.gradient_scaler.step(self.optimizer)
                    self.gradient_scaler.update()
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.config.debug:
                    for param in self.optimizer.param_groups[1]["params"]:
                        break
                    logging.debug(param.data.view(-1)[0])

            self.lr_scheduler.step()

            # accumulate outputs
            accumulator(
                {
                    "average_needle_heatmap_value": mean_predictions_in_needle,
                    "average_prostate_heatmap_value": mean_predictions_in_prostate,
                    **batch,
                }
            )

            # log metrics
            step_metrics = {
                "train_loss": loss.item() / self.config.accumulate_grad_steps
            }
            encoder_lr = self.optimizer.param_groups[0]["lr"]
            main_lr = self.optimizer.param_groups[1]["lr"]
            cnn_lr = self.optimizer.param_groups[2]["lr"]
            step_metrics["encoder_lr"] = encoder_lr
            step_metrics["main_lr"] = main_lr
            step_metrics["cnn_lr"] = cnn_lr

            wandb.log(step_metrics)

            # log images
            if train_iter % 100 == 0 and self.config.log_images:
                self.show_example(batch_for_image_generation)
                wandb.log({f"{desc}_example": wandb.Image(plt)})
                plt.close()

        # compute and log metrics
        results_table = accumulator.compute()
        # results_table.to_csv(os.path.join(self.config.exp_dir, f"{desc}_epoch_{self.epoch}_results.csv"))
        # wandb.save(os.path.join(self.config.exp_dir, f"{desc}_epoch_{self.epoch}_results.csv"))
        return self.create_and_report_metrics(results_table, desc="train")

    @torch.no_grad()
    def run_eval_epoch(self, loader, desc="eval"):
        self.model.eval()

        accumulator = DataFrameCollector()

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            batch_for_image_generation = (
                batch.copy()
            )  # we pop some keys from batch, so we keep a copy for image generation
            bmode = batch.pop("bmode").to(self.config.device)
            needle_mask = batch.pop("needle_mask").to(self.config.device)
            prostate_mask = batch.pop("prostate_mask").to(self.config.device)

            psa = batch["psa"].to(self.config.device)
            age = batch["age"].to(self.config.device)
            family_history = batch["family_history"].to(self.config.device)
            anatomical_location = batch["loc"].to(self.config.device)
            B = len(bmode)
            task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)
            approx_psa_density = batch["approx_psa_density"].to(self.config.device)

            core_ids = batch["core_id"]
            if self.config.custom_prompt_table_path is not None:
                custom_prompts = self.get_custom_prompts(core_ids)
            else: 
                custom_prompts = None

            if 'rf' in batch: 
                rf = batch.pop("rf").to(self.config.device)
            else: 
                rf = None 

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                heatmap_logits = self.model(
                    bmode,
                    task_id=task_id,
                    anatomical_location=anatomical_location,
                    psa=psa,
                    age=age,
                    family_history=family_history,
                    prostate_mask=prostate_mask,
                    needle_mask=needle_mask,
                    approx_psa_density=approx_psa_density,
                    rf_image = rf,
                    custom=custom_prompts
                )

                # compute predictions
                masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
                predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)
                mean_predictions_in_needle = []
                for j in range(B):
                    mean_predictions_in_needle.append(
                        predictions[batch_idx == j].sigmoid().mean()
                    )
                mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)

                prostate_masks = prostate_mask > 0.5
                predictions, batch_idx = MaskedPredictionModule()(
                    heatmap_logits, prostate_masks
                )
                mean_predictions_in_prostate = []
                for j in range(B):
                    mean_predictions_in_prostate.append(
                        predictions[batch_idx == j].sigmoid().mean()
                    )
                mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate)

            if train_iter % 100 == 0 and self.config.log_images:
                self.show_example(batch_for_image_generation)
                wandb.log({f"{desc}_example": wandb.Image(plt)})
                plt.close()

            accumulator(
                {
                    "average_needle_heatmap_value": mean_predictions_in_needle,
                    "average_prostate_heatmap_value": mean_predictions_in_prostate,
                    **batch,
                }
            )

        results_table = accumulator.compute()

        return self.create_and_report_metrics(results_table, desc=desc)

    def create_and_report_metrics(self, results_table, desc="eval"):
        
        # core predictions
        predictions = results_table.average_needle_heatmap_value.values
        labels = results_table.label.values
        involvement = results_table.involvement.values

        core_probs = predictions
        core_labels = labels

        metrics = {}
        metrics_ = calculate_metrics(
            predictions, labels, log_images=self.config.log_images
        )
        metrics.update(metrics_)

        # high involvement core predictions
        high_involvement = involvement > 0.4
        benign = core_labels == 0
        keep = np.logical_or(high_involvement, benign)
        if keep.sum() > 0:
            core_probs = core_probs[keep]
            core_labels = core_labels[keep]
            metrics_ = calculate_metrics(
                core_probs, core_labels, log_images=self.config.log_images
            )
            metrics.update(
                {
                    f"{metric}_high_involvement": value
                    for metric, value in metrics_.items()
                }
            )

        # patient predictions
        predictions = (
            results_table.groupby("patient_id")
            .average_prostate_heatmap_value.mean()
            .values
        )
        labels = (
            results_table.groupby("patient_id").clinically_significant.sum() > 0
        ).values
        metrics_ = calculate_metrics(
            predictions, labels, log_images=self.config.log_images
        )
        metrics.update(
            {f"{metric}_patient": value for metric, value in metrics_.items()}
        )

        metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = self.epoch
        wandb.log(metrics)
        return metrics

    @torch.no_grad()
    def show_example(self, batch):
        # don't log images by default, since they take up a lot of space.
        # should be considered more of a debuagging/demonstration tool
        if self.config.log_images is False:
            return

        bmode = batch["bmode"].to(self.config.device)
        needle_mask = batch["needle_mask"].to(self.config.device)
        prostate_mask = batch["prostate_mask"].to(self.config.device)
        label = batch["label"].to(self.config.device)
        involvement = batch["involvement"].to(self.config.device)
        psa = batch["psa"].to(self.config.device)
        age = batch["age"].to(self.config.device)
        family_history = batch["family_history"].to(self.config.device)
        anatomical_location = batch["loc"].to(self.config.device)
        approx_psa_density = batch["approx_psa_density"].to(self.config.device)
        if 'rf' in batch: 
            rf = batch.pop("rf").to(self.config.device)
        else: 
            rf = None 

        core_ids = batch["core_id"]
        if self.config.custom_prompt_table_path is not None:
            custom_prompts = self.get_custom_prompts(core_ids)
        else: 
            custom_prompts = None

        B = len(bmode)
        task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

        logits = self.model(
            bmode,
            task_id=task_id,
            anatomical_location=anatomical_location,
            psa=psa,
            age=age,
            family_history=family_history,
            prostate_mask=prostate_mask,
            needle_mask=needle_mask,
            approx_psa_density=approx_psa_density,
            custom=custom_prompts,
            rf_image=rf
        )

        pred = logits.sigmoid()

        needle_mask = needle_mask.cpu()
        prostate_mask = prostate_mask.cpu()
        logits = logits.cpu()
        pred = pred.cpu()
        image = bmode.cpu()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        [ax.set_axis_off() for ax in ax.flatten()]
        kwargs = dict(vmin=0, vmax=1, extent=(0, 46, 0, 28))

        ax[0].imshow(image[0].permute(1, 2, 0), **kwargs)
        prostate_mask = prostate_mask.cpu()
        ax[0].imshow(
            prostate_mask[0, 0], alpha=prostate_mask[0][0] * 0.3, cmap="Blues", **kwargs
        )
        ax[0].imshow(needle_mask[0, 0], alpha=needle_mask[0][0], cmap="Reds", **kwargs)
        ax[0].set_title(f"Ground truth label: {label[0].item()}")

        ax[1].imshow(pred[0, 0], **kwargs)

        valid_loss_region = (prostate_mask[0][0] > 0.5).float() * (
            needle_mask[0][0] > 0.5
        ).float()

        alpha = torch.nn.functional.interpolate(
            valid_loss_region[None, None],
            size=(self.config.mask_size, self.config.mask_size),
            mode="nearest",
        )[0, 0]
        ax[2].imshow(pred[0, 0], alpha=alpha, **kwargs)

    def save_experiment_state(self):
        if self.exp_state_path is None:
            return
        logging.info(f"Saving experiment snapshot to {self.exp_state_path}")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
                "gradient_scaler": self.gradient_scaler.state_dict(),
                "rng": get_all_rng_states(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            },
            self.exp_state_path,
        )

    def save_model_weights(self, score, is_best_score=False):
        if self.config.checkpoint_dir is None:
            return

        if self.config.save_weights == "best":
            if not is_best_score:
                return
            else:
                fname = "best_model.ckpt"
        else:
            fname = f"model_epoch{self.epoch}_auc{score:.2f}.ckpt"

        logging.info("Saving model to checkpoint directory")
        logging.info(f"Checkpoint directory: {self.config.checkpoint_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.checkpoint_dir, fname),
        )
        # save config as json file
        with open(
            os.path.join(
                self.config.checkpoint_dir,
                f"config.json",
            ),
            "w",
        ) as f:
            

            json.dump(vars(self.config), f)

    def teardown(self):
        # remove experiment state file
        if self.exp_state_path is not None:
            os.remove(self.exp_state_path)


class MaskedPredictionModule(nn.Module):
    """
    Computes the patch and core predictions and labels within the valid loss region for a heatmap.
    """

    def __init__(self):
        super().__init__()

    def forward(self, heatmap_logits, mask):
        """Computes the patch and core predictions and labels within the valid loss region."""
        B, C, H, W = heatmap_logits.shape

        assert mask.shape == (
            B,
            1,
            H,
            W,
        ), f"Expected mask shape to be {(B, 1, H, W)}, got {mask.shape} instead."

        # mask = mask.float()
        # mask = torch.nn.functional.interpolate(mask, size=(H, W)) > 0.5

        core_idx = torch.arange(B, device=heatmap_logits.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[..., 0]
        logits_flattened = rearrange(heatmap_logits, "b c h w -> (b h w) c", h=H, w=W)

        logits = logits_flattened[mask_flattened]
        core_idx = core_idx_flattened[mask_flattened]

        patch_logits = logits

        return patch_logits, core_idx


def involvement_tolerant_loss(patch_logits, patch_labels, core_indices, involvement):
    batch_size = len(involvement)
    loss = torch.tensor(0, dtype=torch.float32, device=patch_logits.device)
    for i in range(batch_size):
        patch_logits_for_core = patch_logits[core_indices == i]
        patch_labels_for_core = patch_labels[core_indices == i]
        involvement_for_core = involvement[i]
        if patch_labels_for_core[0].item() == 0:
            # core is benign, so label noise is assumed to be low
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_logits_for_core, patch_labels_for_core
            )
        elif involvement_for_core.item() > 0.65:
            # core is high involvement, so label noise is assumed to be low
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_logits_for_core, patch_labels_for_core
            )
        else:
            # core is of intermediate involvement, so label noise is assumed to be high.
            # we should be tolerant of the model's "false positives" in this case.
            pred_index_sorted_by_cancer_score = torch.argsort(
                patch_logits_for_core[:, 0], descending=True
            )
            patch_logits_for_core = patch_logits_for_core[
                pred_index_sorted_by_cancer_score
            ]
            patch_labels_for_core = patch_labels_for_core[
                pred_index_sorted_by_cancer_score
            ]
            n_predictions = patch_logits_for_core.shape[0]
            patch_predictions_for_core_for_loss = patch_logits_for_core[
                : int(n_predictions * involvement_for_core.item())
            ]
            patch_labels_for_core_for_loss = patch_labels_for_core[
                : int(n_predictions * involvement_for_core.item())
            ]
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_predictions_for_core_for_loss,
                patch_labels_for_core_for_loss,
            )


def simple_mil_loss(
    patch_logits,
    patch_labels,
    core_indices,
    top_percentile=0.2,
    pos_weight=torch.tensor(1.0),
):
    ce_loss = nn.functional.binary_cross_entropy_with_logits(
        patch_logits, patch_labels, pos_weight=pos_weight, reduction="none"
    )

    loss = torch.tensor(0, dtype=torch.float32, device=patch_logits.device)

    for i in torch.unique(core_indices):
        patch_losses_for_core = ce_loss[core_indices == i]
        n_patches = len(patch_losses_for_core)
        n_patches_to_keep = int(n_patches * top_percentile)
        patch_losses_for_core_sorted = torch.sort(patch_losses_for_core)[0]
        patch_losses_for_core_to_keep = patch_losses_for_core_sorted[:n_patches_to_keep]
        loss += patch_losses_for_core_to_keep.mean()

    return loss


class CancerDetectionLossBase(nn.Module):
    """Loss to be computed based on pixel-level logits, prostate mask, needle mask, label and involvement"""
    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        raise NotImplementedError


class CancerDetectionValidRegionLoss(CancerDetectionLossBase):
    def __init__(
        self,
        base_loss: str = "ce",
        loss_pos_weight: float = 1.0,
        prostate_mask: bool = True,
        needle_mask: bool = True,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask

    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            if self.prostate_mask:
                mask &= prostate_mask[i] > 0.5
            if self.needle_mask:
                mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        labels = torch.zeros(len(predictions), device=predictions.device)
        for i in range(len(predictions)):
            labels[i] = label[batch_idx[i]]
        labels = labels[..., None]  # needs to match N, C shape of preds

        loss = torch.tensor(0, dtype=torch.float32, device=predictions.device)
        if self.base_loss == "ce":
            loss += nn.functional.binary_cross_entropy_with_logits(
                predictions,
                labels,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=predictions.device
                ),
            )
        elif self.base_loss == "gce":
            # we should convert to "two class" classification problem
            loss_fn = BinaryGeneralizedCrossEntropy()
            loss += loss_fn(predictions, labels)
        elif self.base_loss == "mae":
            loss_unreduced = nn.functional.l1_loss(
                predictions, labels, reduction="none"
            )
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()
        else:
            raise ValueError(f"Unknown base loss: {self.base_loss}")

        return loss


class CancerDetectionSoftValidRegionLoss(CancerDetectionLossBase):
    def __init__(
        self,
        loss_pos_weight: float = 1,
        prostate_mask: bool = True,
        needle_mask: bool = True,
        sigma: float = 15,
    ):
        super().__init__()
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask
        self.sigma = sigma

    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        masks = []
        for i in range(len(cancer_logits)):
            mask = prostate_mask[i] > 0.5
            mask = mask & (needle_mask[i] > 0.5)
            mask = mask.float().cpu().numpy()[0]

            # resize and blur mask
            

            mask = resize(mask, (256, 256), order=0)
            

            mask = gaussian(mask, self.sigma, mode="constant", cval=0)
            mask = mask - mask.min()
            mask = mask / mask.max()
            mask = torch.tensor(mask, device=cancer_logits.device)[None, ...]

            masks.append(mask)
        masks = torch.stack(masks)

        B = label.shape[0]
        label = label.repeat(B, 1, 256, 256).float()
        loss_by_pixel = nn.functional.binary_cross_entropy_with_logits(
            cancer_logits,
            label,
            pos_weight=torch.tensor(self.loss_pos_weight, device=cancer_logits.device),
            reduction="none",
        )
        loss = (loss_by_pixel * masks).mean()
        return loss


class MultiTermCanDetLoss(CancerDetectionLossBase):
    def __init__(self, loss_terms: list[CancerDetectionLossBase], weights: list[float]):
        super().__init__()
        self.loss_terms = loss_terms
        self.weights = weights

    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        loss = torch.tensor(0, dtype=torch.float32, device=cancer_logits.device)
        for term, weight in zip(self.loss_terms, self.weights):
            loss += weight * term(
                cancer_logits, prostate_mask, needle_mask, label, involvement
            )
        return loss


class ProstNFound(nn.Module):
    PROMPT_OPTIONS = [
        "task",
        "anatomical",
        "psa",
        "age",
        "family_history",
        "prostate_mask",
        "sparse_cnn_patch_features",
        "data_independent_prompts",
        "approx_psa_density",
        "sparse_cnn_patch_features_rf",
        "dense_cnn_features",
        "custom",
    ]

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
        n_tasks=1,
        prompts: list[str] = [],
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
    ):
        super().__init__()
        self.prompts = prompts
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

        for p in prompts:
            if not p in self.PROMPT_OPTIONS:
                raise ValueError(
                    f"Unknown prompt option: {p}. Options are {self.PROMPT_OPTIONS}"
                )

        

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

        # BUILD PROMPT MODULES
        EMBEDDING_DIM = 256

        # null prompt - used for prompt dropout
        self.null_prompt = nn.Parameter(torch.zeros(1, EMBEDDING_DIM))

        # used for multitask training, but not currently used
        self.task_prompt_module = nn.Embedding(n_tasks, EMBEDDING_DIM)

        # 6 anatomical locations (mid-lateral, mid-medial, apex-lateral, apex-medial, base-lateral, base-medial)
        self.anatomical_prompt_module = nn.Embedding(6, EMBEDDING_DIM)

        # embed floating point values to 256 dim
        self.psa_prompt_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM),
        )
        self.age_prompt_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM),
        )
        self.approx_psa_density_prompt_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM),
        )
        self.custom_prompt_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM),
        )

        # 3 values for family history: 0, 1, 2 (yes, no, unknown)
        self.family_history_prompt_module = nn.Embedding(3, EMBEDDING_DIM)

        # CNN for extracting patch features
       

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
        
        
        self.patch_aggregator = TransformerEncoder(
            n_layers=6, n_heads=8, d_model=256, d_feed_forward=256, dropout=0.1
        )

        self.dense_feature_projection = nn.Conv2d(512, EMBEDDING_DIM, kernel_size=1)

        # project the CNN features to the prompt space
        #self.patch_feature_prompt_module = nn.Linear(512, EMBEDDING_DIM)
        self.patch_feature_prompt_module = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM),
        )

        self.pad_token = nn.Parameter(
            torch.zeros(EMBEDDING_DIM)
        )  # for padding the number of patches to a fixed number in a minibatch

        # data independent prompts
        self.data_independent_prompts = nn.Parameter(torch.randn(1, 10, EMBEDDING_DIM))

    def forward(
        self,
        image=None,
        task_id=None,
        anatomical_location=None,
        psa=None,
        age=None,
        family_history=None,
        prostate_mask=None,
        needle_mask=None,
        approx_psa_density=None,
        rf_image=None,
        custom=None,
        return_prompt_embeddings=False,
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

        if "prostate_mask" in self.prompts:
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

        if "dense_cnn_features" in self.prompts:
            dense_features = self.patch_feature_cnn[0](image)
            dense_features = self.patch_feature_cnn[1].forward_features(dense_features)
            dense_features = self.dense_feature_projection(dense_features)
            dense_features = torch.nn.functional.interpolate(
                dense_features, size=dense_embedding.shape[-2:]
            )
            if self.training: 
                dense_features = torch.nn.functional.dropout(dense_features, p=0.5, training=True)
            dense_embedding = dense_embedding + dense_features

        sparse_embedding = sparse_embedding.repeat_interleave(len(image), 0)

        if "task" in self.prompts:
            task_embedding = self.task_prompt_module(task_id)
            task_embedding = task_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, task_embedding], dim=1)

        if "anatomical" in self.prompts:
            if (
                anatomical_location is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                anatomical_embedding = self.null_prompt.repeat_interleave(
                    len(task_id), 0
                )
            else:
                anatomical_embedding = self.anatomical_prompt_module(
                    anatomical_location
                )
            anatomical_embedding = anatomical_embedding[:, None, :]
            sparse_embedding = torch.cat(
                [sparse_embedding, anatomical_embedding], dim=1
            )

        if "psa" in self.prompts:
            if (
                psa is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                psa_embedding = self.null_prompt.repeat_interleave(len(task_id), 0)
            else:
                psa_embedding = self.psa_prompt_module(psa)
            psa_embedding = psa_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, psa_embedding], dim=1)

        if "approx_psa_density" in self.prompts:
            if (
                psa is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                approx_psa_density_embedding = self.null_prompt.repeat_interleave(
                    len(task_id), 0
                )
            else:
                approx_psa_density_embedding = self.approx_psa_density_prompt_module(
                    approx_psa_density
                )
            approx_psa_density_embedding = approx_psa_density_embedding[:, None, :]
            sparse_embedding = torch.cat(
                [sparse_embedding, approx_psa_density_embedding], dim=1
            )

        if "age" in self.prompts:
            if (
                age is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                age_embedding = self.null_prompt.repeat_interleave(len(task_id), 0)
            else:
                age_embedding = self.age_prompt_module(age)
            age_embedding = age_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, age_embedding], dim=1)

        if "family_history" in self.prompts:
            if (
                family_history is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                family_history = torch.ones_like(task_id) * 2  # this encodes "unknown"
            family_history_embedding = self.family_history_prompt_module(family_history)
            family_history_embedding = family_history_embedding[:, None, :]
            sparse_embedding = torch.cat(
                [sparse_embedding, family_history_embedding], dim=1
            )

        if "data_independent_prompts" in self.prompts:
            sparse_embedding = torch.cat(
                [
                    sparse_embedding,
                    self.data_independent_prompts.repeat_interleave(B, 0),
                ],
                dim=1,
            )

        if "custom" in self.prompts:
            if (
                custom is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                custom_embedding = self.null_prompt.repeat_interleave(len(task_id), 0)
            else:
                custom_embedding = self.custom_prompt_module(custom)
            custom_embedding = custom_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, custom_embedding], dim=1)

        if "sparse_cnn_patch_features" in self.prompts:
            # we need to extract patches from the images.
            patch_cnn_sparse_embeddings = self.get_cnn_patch_embedding_bmode(
                image, needle_mask, prostate_mask
            )
            if patch_cnn_sparse_embeddings is not None:
                sparse_embedding = torch.cat(
                    [sparse_embedding, patch_cnn_sparse_embeddings], dim=1
                )

        if "sparse_cnn_patch_features_rf" in self.prompts:
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
                patch_embeddings_for_batch = torch.mean(patch_embeddings_for_batch, dim=0, keepdim=True)
            elif self.pool_patch_features == "max":
                if len(patch_embeddings_for_batch) == 0:
                    return None
                patch_embeddings_for_batch = torch.max(patch_embeddings_for_batch, dim=0, keepdim=True).values
            sparse_embeddings_by_batch.append(patch_embeddings_for_batch)

        max_len = max([len(e) for e in sparse_embeddings_by_batch])
        patch_cnn_sparse_embeddings = torch.zeros(B, max_len, 256, device=DEVICE)
        for i, e in enumerate(sparse_embeddings_by_batch):
            patch_cnn_sparse_embeddings[i, : len(e)] = e
            patch_cnn_sparse_embeddings[i, len(e) :] = self.pad_token[None, None, :]

        if self.prompt_dropout > 0 and self.training:
            for i in range(patch_cnn_sparse_embeddings.shape[1]):
                if torch.rand(1) < self.prompt_dropout:
                    patch_cnn_sparse_embeddings[
                        :, i, :
                    ] = self.null_prompt.repeat_interleave(B, 0)

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
                align_to="topright"
            )
            for position, patch in zip(pv.positions, pv):
                patches.append(torch.from_numpy(patch).permute(2, 0, 1))
                positions.append(torch.from_numpy(position))
                batch_indices.append(i)

        logging.debug(f"Extracted {len(patches)} patches from {B} rf images")
        if len(patches) == 0: 
            return None

        patches = torch.stack(patches).to(DEVICE)
        # patches should be resized to 256 by 256 as used in the RF CNNs
        patches = torch.nn.functional.interpolate(patches, size=(256, 256), mode='bilinear')

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
            patch_embeddings_for_batch = patch_cnn_output[batch_indices == i] # N x 256
            if self.pool_patch_features == 'mean': 
                if len(patch_embeddings_for_batch) == 0:
                    return None # no patches found
                patch_embeddings_for_batch = torch.mean(patch_embeddings_for_batch, dim=0, keepdim=True)
            elif self.pool_patch_features == 'max':
                if len(patch_embeddings_for_batch) == 0:
                    return None
                patch_embeddings_for_batch = torch.max(patch_embeddings_for_batch, dim=0, keepdim=True).values
            sparse_embeddings_by_batch.append(patch_embeddings_for_batch)

        max_len = max([len(e) for e in sparse_embeddings_by_batch])
        patch_cnn_sparse_embeddings = torch.zeros(B, max_len, 256, device=DEVICE)
        for i, e in enumerate(sparse_embeddings_by_batch):
            patch_cnn_sparse_embeddings[i, : len(e)] = e
            patch_cnn_sparse_embeddings[i, len(e) :] = self.pad_token[None, None, :]

        if self.prompt_dropout > 0 and self.training:
            for i in range(patch_cnn_sparse_embeddings.shape[1]):
                if torch.rand(1) < self.prompt_dropout:
                    patch_cnn_sparse_embeddings[
                        :, i, :
                    ] = self.null_prompt.repeat_interleave(B, 0)

        B, N, C = patch_cnn_sparse_embeddings.shape
        if self.pool_patch_features == 'transformer': 
            patch_cnn_sparse_embeddings = self.patch_aggregator(
                patch_cnn_sparse_embeddings
            )
            patch_cnn_sparse_embeddings = patch_cnn_sparse_embeddings.mean(dim=1, keepdim=True)

        return patch_cnn_sparse_embeddings

    def train(self, mode: bool = True):
        super().train(mode)
        if (
            self.sparse_cnn_backbone_path is not None
            and self.patch_feature_cnn is not None
        ):
            self.patch_feature_cnn.eval()

    def get_params_groups(self):
        
        encoder_parameters = [
            p
            for (k, p) in self.medsam_model.image_encoder.named_parameters()
            if "neck" not in k
        ]
        warmup_parameters = chain(
            self.medsam_model.mask_decoder.parameters(),
            self.task_prompt_module.parameters(),
            self.anatomical_prompt_module.parameters(),
            self.psa_prompt_module.parameters(),
            self.age_prompt_module.parameters(),
            [self.null_prompt],
            [self.data_independent_prompts],
            self.family_history_prompt_module.parameters(),
            self.approx_psa_density_prompt_module.parameters(),
            self.patch_feature_prompt_module.parameters(),
            self.custom_prompt_module.parameters(),
            self.medsam_model.image_encoder.neck.parameters(),
            self.medsam_model.prompt_encoder.parameters(),
            [self.pad_token],
            self.dense_feature_projection.parameters(),
        )
        cnn_parameters = (
            self.patch_feature_cnn.parameters()
            if self.patch_feature_cnn is not None
            else []
        )

        return encoder_parameters, warmup_parameters, cnn_parameters

    @classmethod
    def add_arguments(cls, parser: ArgumentParser):
        group = parser.add_argument_group("Model")
        group.add_argument(
            "--backbone",
            type=str,
            choices=cls.BACKBONE_OPTIONS,
            default="medsam",
            help="The backbone to use for the model.",
        )
        group.add_argument(
            "--prompts",
            type=str,
            nargs="+",
            default=["task", "anatomical", "psa", "age", "family_history"],
            help="The prompts to use for the model.",
            choices=cls.PROMPT_OPTIONS + ["none"],
        )
        group.add_argument(
            "--prompt_dropout",
            type=float,
            default=0.0,
            help="The dropout to use for the prompts.",
        )
        group.add_argument(
            "--cnn_mode",
            choices=("dense_prompt", "sparse_prompt", "disabled"),
            type=lambda x: None if x == "disabled" else str(x),
            help="Mode to use for the CNN branch.",
            default="disabled",
        )
        group.add_argument(
            "--replace_patch_embed",
            action="store_true",
            help="If True, replaces the patch embedding with a learned convolutional patch embedding.",
        )
        group.add_argument(
            "--sparse_cnn_backbone_path",
            type=str,
            default=None,
            help="The path to the sparse CNN backbone to use. If None, randomly initializes and trains the backbone.",
        )
        group.add_argument(
            "--freeze_mask_decoder",
            action="store_true",
            help="If True, freezes the mask decoder.",
        )
        group.add_argument(
            "--freeze_image_encoder",
            action="store_true",
            help="If True, freezes the image encoder. This is useful for partial finetuning or prompt tuning",
        )
        group.add_argument(
            "--freeze_cnn",
            action="store_true",
            help="If True, freezes the CNN.",
        )
        group.add_argument('--img_emb_dropout', type=float, default=0.0, help="Dropout for the image embeddings")
        group.add_argument('--cnn_patches_whole_prostate', action='store_true', help="If True, extracts patches from the whole prostate instead of just the needle mask")
        group.add_argument('--no_pos_embed_cnn_patch', action='store_true', help="If True, disables positional encoding for the CNN patch embeddings")
        group.add_argument('--pool_patch_features', type=str, default='mean', help="the pooling method to use for the patch features. Can be 'mean' or 'max' or 'none'")
        return group

    @staticmethod
    def from_args(args):
        if "none" in args.prompts:
            args.prompts = []

        print(
            f"""Building model with args: 
    {args.backbone=}
    {args.prompts=}
    {args.prompt_dropout=}
    {args.sparse_cnn_backbone_path=}
    {args.replace_patch_embed=}
"""
        )

        return ProstNFound(
            sam_backbone=args.backbone,
            prompts=args.prompts,
            prompt_dropout=args.prompt_dropout,
            sparse_cnn_backbone_path=args.sparse_cnn_backbone_path,
            replace_patch_embed=args.replace_patch_embed,
            freeze_mask_decoder=args.freeze_mask_decoder,
            freeze_image_encoder=args.freeze_image_encoder,
            freeze_cnn=args.freeze_cnn,
            img_emb_dropout=args.img_emb_dropout,
            cnn_patches_whole_prostate=args.cnn_patches_whole_prostate,
            pos_embed_cnn_patch=not args.no_pos_embed_cnn_patch,
            pool_patch_features=args.pool_patch_features,
        )


class BinaryGeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, pred, labels):
        pred = pred.sigmoid()[..., 0]
        labels = labels[..., 0].long()
        pred = torch.stack([1 - pred, pred], dim=-1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, 2).float().to(pred.device)
        gce = (1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()


def load_encoder_weights(image_encoder, weights_path, adapter_mode=None):
    raise NotImplementedError(
        "Loading encoder from different modules is not currently supported."
    )

    state = torch.load(weights_path, map_location="cpu")
    if adapter_mode is None:
        image_encoder.load_state_dict(state)
    elif "dino" in adapter_mode:
        from train_medsam_dino_style import MedSAMDino

        model = MedSAMDino()
        model.load_state_dict(state)
        image_encoder_state = model.image_encoder.state_dict()
        image_encoder.load_state_dict(image_encoder_state)
    elif "ibot" in adapter_mode:
        from train_medsam_ibot_style import MedSAMIBot

        model = MedSAMIBot(8192, 8192)
        model.load_state_dict(state)
        image_encoder_state = model.image_encoder.state_dict()
        image_encoder.load_state_dict(image_encoder_state)
    else:
        raise ValueError(f"Unknown adapter mode: {adapter_mode}")


if __name__ == "__main__":

    args = parse_args()
    experiment = Experiment(args)
    experiment.run()

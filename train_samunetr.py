"""
Our implementation of SAM_UNETR: 

@article{alzate2023sam,
  title={SAM-UNETR: Clinically Significant Prostate Cancer Segmentation Using Transfer Learning From Large Model},
  author={Alzate-Grisales, Jesus Alejandro and Mora-Rubio, Alejandro and Garc{\'\i}a-Garc{\'\i}a, Francisco and Tabares-Soto, Reinel and De La Iglesia-Vay{\'a}, Maria},
  journal={IEEE Access},
  volume={11},
  pages={118217--118228},
  year={2023},
  publisher={IEEE}
}

For prostate cancer detection from Micro-Ultrasound.
"""

from dataclasses import dataclass, field
import logging
import os
import typing as tp
from abc import ABC, abstractmethod
import numpy as np
from simple_parsing import parse
from simple_parsing.helpers.serialization.serializable import Serializable
import torch
import torch.nn as nn
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from src.utils import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

import wandb
from src.data_factory import DataLoaderFactory
import argparse
from rich_argparse import ArgumentDefaultsRichHelpFormatter
from torch.optim import AdamW
from src.utils import DataFrameCollector, calculate_metrics, cosine_scheduler
from skimage.transform import resize
from skimage.filters import gaussian
from src.sam_wrappers import build_medsam, build_sam
from src.unetr import UNETR
from src.dataset import get_dataloaders_main
import config
from src.losses import LossOptions, build_loss


PROMPT_OPTIONS = [
    "task",
    "anatomical",
    "psa",
    "age",
    "family_history",
    "prostate_mask",
    "dense_cnn_image_features",
    "sparse_cnn_maskpool_features_needle",
    "sparse_cnn_maskpool_features_prostate",
    "sparse_cnn_patch_features",
]


# fmt: off
@dataclass
class Args(Serializable):
    """Configuration for sam-Unetr training"""

    splits_json_path: str = "splits.json"
    paths: config.DataPaths = config.DataPaths()
    data: config.MainDataOptions = config.MainDataOptions()
    wandb: config.WandbOptions = field(default_factory=config.WandbOptions)
    loss: LossOptions = LossOptions()

    # training
    optimizer: str = "adamw"
    lr: float = 1e-5 
    encoder_lr: float = 1e-5
    warmup_lr: float = 1e-4
    warmup_epochs: int = 5
    wd: float = 0 # The weight decay to use for training. We found weight decay can degrade performance (likely due to forgetting foundation model pretraining) so it is off by default.
    epochs: int = 30 # The number of epochs for the training and learning rate annealing.
    cutoff_epoch: int = None # If not None, the training will stop after this epoch, but this will not affect the learning rate scheduler.
    accumulate_grad_steps: int = 8 # The number of gradient accumulation steps to use. Can be used to increase the effective batch size when GPU memory is limited.
    run_test: bool = False # If True, runs the test set. Should disable for experiments related to model selection (e.g. hyperparameter tuning)
    test_every_epoch: bool = False # Only used if `--run_test` is set. If this is set, runs the test set every epoch. Otherwise, only runs it when a new best validation score is achieved.

    # model 
    backbone: str = 'medsam'

    # miscellaneous
    encoder_weights_path: str = None # The path to the encoder weights to use. If None, uses the Foundation Model initialization
    encoder_load_mode: str = "none" # The mode to use for loading the encoder weights.
    seed: int = 42 # The seed to use for training.
    use_amp: bool = False # If True, uses automatic mixed precision.
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' # The device to use for training
    exp_dir: str = "experiments/default" # The directory to use for the experiment.
    checkpoint_dir: str = None # The directory to use for the checkpoints. If None, does not save checkpoints.
    debug: bool = False # If True, runs in debug mode.
    save_weights: str = "best" # The mode to use for saving weights.
# fmt: on


def parse_args():
    # fmt: off
    
    parser = argparse.ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsRichHelpFormatter)

    # group = parser.add_argument_group("Data", "Arguments related to data loading and preprocessing")
    # group.add_argument("--fold", type=int, default=None, help="The fold to use. If not specified, uses leave-one-center-out cross-validation.")
    # group.add_argument("--n_folds", type=int, default=None, help="The number of folds to use for cross-validation.")
    # group.add_argument("--test_center", type=str, default=None, 
    #                     help="If not None, uses leave-one-center-out cross-validation with the specified center as test. If None, uses k-fold cross-validation.")
    # group.add_argument("--val_seed", type=int, default=0, 
    #                    help="The seed to use for validation split.")            
    # group.add_argument("--undersample_benign_ratio", type=lambda x: float(x) if not x.lower() == 'none' else None, default=None,
    #                    help="""If not None, undersamples benign cores with the specified ratio.""")
    # group.add_argument("--min_involvement_train", type=float, default=0.0,
    #                    help="""The minimum involvement threshold to use for training.""")
    # group.add_argument("--batch_size", type=int, default=1, help="The batch size to use for training.")
    # group.add_argument("--augmentations", type=str, default="translate", help="The augmentations to use for training.")
    # group.add_argument("--remove_benign_cores_from_positive_patients", action="store_true", help="If True, removes benign cores from positive patients (training only).")

    # group = parser.add_argument_group("Training", "Arguments related to training.")
    # group.add_argument("--optimizer", type=str, default="adamw", help="The optimizer to use for training.")
    # group.add_argument("--lr", type=float, default=1e-5, help="LR for the model.")
    # group.add_argument("--encoder_lr", type=float, default=1e-5, help="LR for the encoder part of the model.")
    # group.add_argument("--warmup_lr", type=float, default=1e-4, help="LR for the warmup, frozen encoder part.")
    # group.add_argument("--warmup_epochs", type=int, default=5, help="The number of epochs to train the warmup, frozen encoder part.")
    # group.add_argument("--wd", type=float, default=0, help="The weight decay to use for training.")
    # group.add_argument("--epochs", type=int, default=30, help="The number of epochs to train for in terms of LR scheduler.")
    # group.add_argument("--cutoff_epoch", type=int, default=None, help="If not None, the training will stop after this epoch, but this will not affect the learning rate scheduler.")
    # group.add_argument("--test_every_epoch", action="store_true", help="If True, runs the test set every epoch.")
    # group.add_argument("--accumulate_grad_steps", type=int, default=8, help="The number of gradient accumulation steps to use.")

    # MODEL
    # group = parser.add_argument_group("Model", "Arguments related to the model.")
    # group.add_argument("--backbone", type=str, choices=('sam', 'medsam'), default='sam', help="The backbone to use for the model.")

    # LOSS
    # parser.add_argument("--n_loss_terms", type=int, default=1, help="The number of loss terms to use.")
    # args, _ = parser.parse_known_args()
    # n_loss_terms = args.n_loss_terms
    # for i in range(n_loss_terms):
    #     group = parser.add_argument_group(f"Loss term {i}", f"Arguments related to loss term {i}.")
    #     group.add_argument(f"--loss_{i}_name", type=str, default="valid_region", choices=('valid_region',), help="The name of the loss function to use."),
    #     group.add_argument(f"--loss_{i}_base_loss_name", type=str, default="ce", 
    #                        choices=('ce', 'gce', 'mae', 'mil'), help="The name of the lower-level loss function to use.")
    #     def str2bool(str): 
    #         return True if str.lower() == 'true' else False
    #     group.add_argument(f"--loss_{i}_pos_weight", type=float, default=1.0, help="The positive class weight for the loss function.")
    #     group.add_argument(f"--loss_{i}_prostate_mask", type=str2bool, default=True, help="If True, the loss will only be applied inside the prostate mask.")
    #     group.add_argument(f"--loss_{i}_needle_mask", type=str2bool, default=True, help="If True, the loss will only be applied inside the needle mask.")
    #     group.add_argument(f"--loss_{i}_weight", type=float, default=1.0, help="The weight to use for the loss function.")

    # group = parser.add_argument_group("Wandb", "Arguments related to wandb.")
    # group.add_argument("--project", type=str, default="miccai2024", help="The wandb project to use.")
    # group.add_argument("--group", type=str, default=None, help="The wandb group to use.")
    # group.add_argument("--name", type=str, default=None, help="The wandb name to use.")
    # group.add_argument("--log_images", action="store_true", help="If True, logs images to wandb.")

    group = parser.add_argument_group("Misc", "Miscellaneous arguments.")
    group.add_argument("--encoder_weights_path", type=str, default=None, help="The path to the encoder weights to use.")
    group.add_argument("--encoder_load_mode", type=str, default="none", choices=("dino_medsam", "ibot_medsam", "image_encoder", "none"), help="The mode to use for loading the encoder weights.")
    group.add_argument("--seed", type=int, default=42, help="The seed to use for training.")
    group.add_argument("--use_amp", action="store_true", help="If True, uses automatic mixed precision.")
    group.add_argument("--device", type=str, default='auto', help="The device to use for training. If 'auto', uses cuda if available, otherwise cpu.")
    group.add_argument("--exp_dir", type=str, default="experiments/default", help="The directory to use for the experiment.")
    group.add_argument("--checkpoint_dir", type=str, default=None, help="The directory to use for the checkpoints. If None, does not save checkpoints.")
    group.add_argument("--debug", action="store_true", help="If True, runs in debug mode.")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit')

    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


class Experiment:
    def __init__(self, config: Args):
        self.config = config

    def setup(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logging.info("Setting up experiment")
        os.makedirs(self.config.exp_dir, exist_ok=True)
        logging.info("Running in directory: " + self.config.exp_dir)

        if self.config.debug:
            self.config.wandb.name = "debug"
        wandb.init(
            project=self.config.wandb.project,
            group=self.config.wandb.group,
            name=self.config.wandb.name,
            config=self.config.to_dict(),
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

        self.model = SAM_UNETR_Wrapper(backbone=self.config.backbone)
        self.model.to(self.config.device)
        torch.compile(self.model)
        self.model.freeze_backbone()  # freeze backbone for first few epochs

        # setup criterion
        self.loss_fn = build_loss(self.config.loss)

    def setup_optimizer(self):

        params = [
            {
                "params": self.model.get_encoder_parameters(),
                "lr": self.config.encoder_lr,
            },
            {
                "params": self.model.get_non_encoder_parameters(),
                "lr": self.config.lr,
            },
        ]
        self.optimizer = AdamW(params, lr=self.config.lr, weight_decay=self.config.wd)

        self.lr_scheduler = cosine_scheduler(
            self.config.lr,
            final_value=0,
            epochs=self.config.epochs,
            warmup_epochs=5,
            niter_per_ep=len(self.train_loader),
            start_warmup_value=0,
        )

        self.warmup_optimizer = AdamW(
            self.model.get_non_encoder_parameters(),
            lr=self.config.warmup_lr,
            weight_decay=self.config.wd,
        )

    def setup_data(self):
        logging.info("Setting up data")

        self.image_size, self.mask_size = 1024, 256

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders_main(
            self.config.paths.data_h5_path,
            self.config.paths.metadata_csv_path,
            self.config.splits_json_path,
            prompts_csv_file=None,
            augment=self.config.data.augmentations,
            image_size=self.config.data.image_size,
            mask_size=self.config.data.mask_size,
            include_rf=False,
            rf_as_bmode=False,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
        )
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

            if new_record or self.config.test_every_epoch:
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

        # if we are in the warmup stage, we use the warmup optimizer
        # and freeze the backbone
        if self.epoch < self.config.warmup_epochs:
            optimizer = self.warmup_optimizer
            stage = "warmup"
        else:
            self.model.unfreeze_backbone()
            optimizer = self.optimizer
            stage = "main"

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            batch_for_image_generation = (
                batch.copy()
            )  # we pop some keys from batch, so we keep a copy for image generation
            if self.config.debug and train_iter > 10:
                break

            # If we are in the main training stage, we update the learning rate
            if stage == "main":
                iteration = train_iter + len(loader) * (
                    self.epoch - self.config.warmup_epochs
                )
                cur_lr = self.lr_scheduler[iteration]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = cur_lr

            # extracting relevant data from the batch
            bmode = batch.pop("bmode").to(self.config.device)
            needle_mask = batch.pop("needle_mask").to(self.config.device)
            prostate_mask = batch.pop("prostate_mask").to(self.config.device)

            label = batch["label"].to(self.config.device)
            involvement = batch["involvement"].to(self.config.device)
            B = len(bmode)
            task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

            # run the model
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # forward pass
                heatmap_logits = self.model(
                    bmode,
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
                self.gradient_scaler.scale(loss).backward()
            else:
                loss.backward()

            # gradient accumulation and optimizer step
            if (train_iter + 1) % self.config.accumulate_grad_steps == 0:
                if self.config.use_amp:
                    self.gradient_scaler.step(optimizer)
                    self.gradient_scaler.update()
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

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
            step_metrics["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(step_metrics)

            # log images
            if train_iter % 100 == 0 and self.config.wandb.log_images:
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

            B = len(bmode)
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                heatmap_logits = self.model(
                    bmode,
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

            if train_iter % 100 == 0 and self.config.wandb.log_images:
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
            predictions, labels, log_images=self.config.wandb.log_images
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
                core_probs, core_labels, log_images=self.config.wandb.log_images
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
            predictions, labels, log_images=self.config.wandb.log_images
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
        if self.config.wandb.log_images is False:
            return

        bmode = batch["bmode"].to(self.config.device)
        needle_mask = batch["needle_mask"].to(self.config.device)
        prostate_mask = batch["prostate_mask"].to(self.config.device)
        label = batch["label"].to(self.config.device)
        B = len(bmode)
        task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

        logits = self.model(
            bmode,
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
            size=(self.config.data.mask_size, self.config.data.mask_size),
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
            },
            self.exp_state_path,
        )

    def save_model_weights(self, score, is_best_score=False):
        if self.config.checkpoint_dir is None or not is_best_score:
            return
        logging.info("Saving model to checkpoint directory")
        logging.info(f"Checkpoint directory: {self.config.checkpoint_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.config.checkpoint_dir,
                f"best_model_epoch{self.epoch}_auc{score:.2f}.ckpt",
            ),
        )

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


class CancerDetectionMILRegionLoss(nn.Module): ...


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


CORE_LOCATIONS = [
    "LML",
    "RBL",
    "LMM",
    "RMM",
    "LBL",
    "LAM",
    "RAM",
    "RML",
    "LBM",
    "RAL",
    "RBM",
    "LAL",
]


class ModelInterface(nn.Module, ABC):
    @abstractmethod
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
    ):
        """Returns the model's heatmap logits."""

    @abstractmethod
    def get_encoder_parameters(self):
        """Returns the parameters of the encoder (backbone)."""

    @abstractmethod
    def get_non_encoder_parameters(self):
        """Returns the parameters of the non-encoder part of the model."""

    def freeze_backbone(self):
        """Freezes the backbone of the model."""
        for param in self.get_encoder_parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes the backbone of the model."""
        for param in self.get_encoder_parameters():
            param.requires_grad = True


class SAM_UNETR_Wrapper(ModelInterface):
    def __init__(self, backbone: tp.Literal["sam", "medsam"] = "sam"):
        super().__init__()

        _sam_model = build_sam() if backbone == "sam" else build_medsam()
        self.model = UNETR(
            _sam_model.image_encoder,
            input_size=1024,
            output_size=256,
        )

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
    ):
        return self.model(image)

    def get_encoder_parameters(self):
        return self.model.image_encoder.parameters()

    def get_non_encoder_parameters(self):
        return [p for k, p in self.model.named_parameters() if "image_encoder" not in k]


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


if __name__ == "__main__":
    args = parse(Args, add_config_path_arg=True)
    print(args.dumps_json())
    experiment = Experiment(args)
    experiment.run()

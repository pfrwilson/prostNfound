from dataclasses import dataclass, field
import logging
import os
import pandas as pd
import numpy as np
from simple_parsing import Serializable, parse
import torch
import wandb
from matplotlib import pyplot as plt
from src.dataset import get_dataloaders_main
from torch.nn import functional as F
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from src.losses import MaskedPredictionModule
from src.utils import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
    DataFrameCollector,
    calculate_metrics,
)
import config
from src.losses import LossOptions, build_loss
from src.prostnfound import ProstNFoundConfig, build_prostnfound


# fmt: off
@dataclass
class Args(Serializable):
    """Configuration for running ProstNFound training"""

    splits_json_path: str = "splits.json"
    paths: config.DataPaths = config.DataPaths()
    data: config.MainDataOptions = config.MainDataOptions()
    model: ProstNFoundConfig = ProstNFoundConfig()
    wandb: config.WandbOptions = field(default_factory=config.WandbOptions)
    loss: LossOptions = LossOptions()

    # training
    optimizer: str = "adamw"
    cnn_lr: float = 1e-5 # The learning rate to use for the CNN. (Only used if sparse_cnn_patch_features of sparse_cnn_patch_features_rf is in the list of prompts.)
    cnn_frozen_epochs: int = 0 # The number of frozen epochs for the cnn.
    cnn_warmup_epochs: int = 5 # The number of linear warmup epochs for the cnn.
    encoder_lr: float = 1e-5 # The learning rate to use for the encoder.
    encoder_frozen_epochs: int = 0 # The number of frozen epochs for the encoder.
    encoder_warmup_epochs: int = 5 # The number of linear warmup epochs for the encoder.
    main_lr: float = 1e-5 # The learning rate to use for the main of the model (mask decoder and patch decoder components)
    main_frozen_epochs: int = 0 # The number of frozen epochs for the main part of the model.
    main_warmup_epochs: int = 5 # The number of linear warmup epochs for the main part of the model.
    wd: float = 0 # The weight decay to use for training. We found weight decay can degrade performance (likely due to forgetting foundation model pretraining) so it is off by default.
    epochs: int = 30 # The number of epochs for the training and learning rate annealing.
    cutoff_epoch: int | None = None # If not None, the training will stop after this epoch, but this will not affect the learning rate scheduler.
    accumulate_grad_steps: int = 8 # The number of gradient accumulation steps to use. Can be used to increase the effective batch size when GPU memory is limited.
    run_test: bool = False # If True, runs the test set. Should disable for experiments related to model selection (e.g. hyperparameter tuning)
    test_every_epoch: bool = False # Only used if `--run_test` is set. If this is set, runs the test set every epoch. Otherwise, only runs it when a new best validation score is achieved.

    # miscellaneous
    encoder_weights_path: str = None # The path to the encoder weights to use. If None, uses the Foundation Model initialization
    encoder_load_mode: str = "none" # The mode to use for loading the encoder weights.
    seed: int = 42 # The seed to use for training.
    use_amp: bool = False # If True, uses automatic mixed precision.
    torch_compile: bool = False # If True, uses torch.compile
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' # The device to use for training
    exp_dir: str = "experiments/default" # The directory to use for the experiment.
    checkpoint_dir: str = None # The directory to use for the checkpoints. If None, does not save checkpoints.
    debug: bool = False # If True, runs in debug mode.
    save_weights: str = "best" # The mode to use for saving weights.
# fmt: on


class Experiment:
    def __init__(self, config: Args):
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
            self.config.wandb.name = "debug"
        wandb.init(
            project=self.config.wandb.project,
            group=self.config.wandb.group,
            name=self.config.wandb.name,
            config=self.config.to_dict(),
            tags=self.config.wandb.tags,
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

        # data
        self.setup_data()

        logging.info("Setting up model")

        # MODEL
        self.model = build_prostnfound(self.config.model)
        self.model.to(self.config.device)

        if self.config.torch_compile:
            torch.compile(self.model)

        self.loss_fn = build_loss(self.config.loss)

        if self.state is not None:
            self.model.load_state_dict(self.state["model"])

        # optimizer
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

    def setup_optimizer(self):

        (
            encoder_parameters,
            warmup_parameters,
            cnn_parameters,
        ) = self.model.get_params_groups()

        params = [
            {"params": encoder_parameters, "lr": self.config.encoder_lr},
            {"params": warmup_parameters, "lr": self.config.main_lr},
            {"params": cnn_parameters, "lr": self.config.cnn_lr},
        ]

        class LRCalculator:
            def __init__(
                self, frozen_epochs, warmup_epochs, total_epochs, niter_per_ep
            ):
                self.frozen_epochs = frozen_epochs
                self.warmup_epochs = warmup_epochs
                self.total_epochs = total_epochs
                self.niter_per_ep = niter_per_ep

            def __call__(self, iter):
                if iter < self.frozen_epochs * self.niter_per_ep:
                    return 0
                elif (
                    iter < (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
                ):
                    return (iter - self.frozen_epochs * self.niter_per_ep) / (
                        self.warmup_epochs * self.niter_per_ep
                    )
                else:
                    cur_iter = (
                        iter
                        - (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
                    )
                    total_iter = (
                        self.total_epochs - self.warmup_epochs - self.frozen_epochs
                    ) * self.niter_per_ep
                    return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))

        self.optimizer = AdamW(params, weight_decay=self.config.wd)

        self.lr_scheduler = LambdaLR(
            self.optimizer,
            [
                LRCalculator(
                    self.config.encoder_frozen_epochs,
                    self.config.encoder_warmup_epochs,
                    self.config.epochs,
                    len(self.train_loader),
                ),
                LRCalculator(
                    self.config.main_frozen_epochs,
                    self.config.main_warmup_epochs,
                    self.config.epochs,
                    len(self.train_loader),
                ),
                LRCalculator(
                    self.config.cnn_frozen_epochs,
                    self.config.cnn_warmup_epochs,
                    self.config.epochs,
                    len(self.train_loader),
                ),
            ],
        )

    def setup_data(self):
        logging.info("Setting up data")

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders_main(
            self.config.paths.data_h5_path,
            self.config.paths.metadata_csv_path,
            self.config.splits_json_path,
            self.config.data.prompt_table_csv_path,
            augment=self.config.data.augmentations,
            image_size=self.config.data.image_size,
            mask_size=self.config.data.mask_size,
            include_rf=self.config.model.use_sparse_cnn_patch_features_rf,
            rf_as_bmode=self.config.data.rf_as_bmode,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
        )

        logging.info(f"Number of training batches: {len(self.train_loader)}")
        logging.info(f"Number of validation batches: {len(self.val_loader)}")
        logging.info(f"Number of test batches: {len(self.test_loader)}")
        logging.info(f"Number of training samples: {len(self.train_loader.dataset)}")
        logging.info(f"Number of validation samples: {len(self.val_loader.dataset)}")
        logging.info(f"Number of test samples: {len(self.test_loader.dataset)}")

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

    def shared_step(self, batch):

        batch = batch.copy()

        # extract relevant data and move to gpu
        bmode = batch.pop("bmode").to(self.config.device)
        needle_mask = batch.pop("needle_mask").to(self.config.device)
        prostate_mask = batch.pop("prostate_mask").to(self.config.device)

        if "rf" in batch:
            rf = batch.pop("rf").to(self.config.device)
        else:
            rf = None

        prompt_keys = (
            self.config.model.floating_point_prompts
            + self.config.model.discrete_prompts
        )
        for key in prompt_keys:
            if key not in batch:
                raise ValueError(
                    f"Prompt key {key} not found in batch. Keys: {list(batch.keys())}"
                )

        prompts = {key: batch[key].to(self.config.device) for key in prompt_keys}

        label = batch["label"].to(self.config.device)
        involvement = batch["involvement"].to(self.config.device)

        B = bmode.shape[0]

        # run the model
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            # forward pass
            heatmap_logits = self.model(
                bmode,
                rf,
                prostate_mask,
                needle_mask,
                return_prompt_embeddings=False,
                **prompts,
            )

            if torch.any(torch.isnan(heatmap_logits)):
                logging.warning("NaNs in heatmap logits")

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

        return (
            loss,
            heatmap_logits,
            mean_predictions_in_needle,
            mean_predictions_in_prostate,
            batch,
        )

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
            #
            (
                loss,
                heatmap_logits,
                mean_predictions_in_needle,
                mean_predictions_in_prostate,
                batch,
            ) = self.shared_step(batch)

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
            if self.config.debug and train_iter > 10:
                break

            batch_for_image_generation = (
                batch.copy()
            )  # we pop some keys from batch, so we keep a copy for image generation

            (
                loss,
                heatmap_logits,
                mean_predictions_in_needle,
                mean_predictions_in_prostate,
                batch,
            ) = self.shared_step(batch)

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

    def create_and_report_metrics(self, results_table: pd.DataFrame, desc="eval"):

        # core predictions
        predictions = results_table.average_needle_heatmap_value.values
        labels = results_table.label.values

        core_probs = predictions
        core_labels = labels

        metrics = {}
        metrics_ = calculate_metrics(
            predictions, labels, log_images=self.config.wandb.log_images
        )
        metrics.update(metrics_)

        # high involvement core predictions
        if "involvement" in results_table.columns:
            involvement = results_table.involvement.values
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
        if "clinically_significant" in results_table.columns:
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

        (
            loss,
            logits,
            mean_predictions_in_needle,
            mean_predictions_in_prostate,
            batch,
        ) = self.shared_step(batch)

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
            self.config.dump_json(f)

    def teardown(self):
        # remove experiment state file
        if self.exp_state_path is not None:
            os.remove(self.exp_state_path)



if __name__ == "__main__":

    args = parse(Args, add_config_path_arg=True)
    print(args.dumps_json())
    experiment = Experiment(args)
    experiment.run()

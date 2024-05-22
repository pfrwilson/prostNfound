"""
Used to train patch-level self-supervised CNN, used to extract prompts for the ProstNFound method
"""

from dataclasses import dataclass
import logging
import os

import numpy as np
from simple_parsing import Serializable, parse, ArgumentParser
import torch
import wandb
from src.patch_model_factory import resnet10t_instance_norm
from torchvision import transforms as T
from tqdm import tqdm

from src.vicreg import VICReg
from src.utils import (
    PatchView, 
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
    cosine_scheduler as make_cosine_scheduler,
)
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.utils import DataFrameCollector
from src.dataset import BModePatchesDataset, RFPatchesDataset
from src.dataset import PatchesTransform as Transform
from src.dataset import PatchesSSLTransform as SSLTransform
import json
from config import DataPaths
import typing as tp 


@dataclass
class Args(Serializable):
    """Run the patch-based self-supervision training for ProstNFound"""

    splits_file: str 
    paths: DataPaths = DataPaths()
    
    data_type: tp.Literal['bmode', 'rf'] = 'bmode' # whether to use BMode or RF data
    patch_size: int = 128 # patch size in pixels (if using bmode) 
    patch_stride: int = 32 # patch stride in pixelse (if using bmode)
    patch_size_mm: tuple[float, float] = (5.0, 5.0) # patch size in mm (if using RF)
    patch_stride_mm: tuple[float, float] = (1.0, 1.0) # patch_size in mm (if using RF)

    batch_size: int = 128
    full_prostate: bool = False # whether to use the full prostate for SSL patches. If False, only select patches within the needle mask.
    epochs: int = 100
    lr: float = 1e-3 
    debug: bool = False
    save_weights_path: str | None = None # Path to save the model weights with the best validation linear probing performance.
    seed: int = 0 # Random seed
    checkpoint_path: str | None = None # Path to save and load experiment state
    outputs_path: str = "outputs.csv" # Path to save outputs of linear probing
    name: str | None = None # Name of the experiment (for wandb)
    run_test: bool = False # If set, computes probing outputs for the test set. Not set by default.


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    return parse(Args, add_config_path_arg=True)


def main(args: Args):
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        state = torch.load(args.checkpoint_path)
    else:
        state = None
    if args.save_weights_path is not None and os.path.exists(args.save_weights_path) and state is None: 
        print(f"Model weights already exist at {args.save_weights_path}. Exiting.")
        return

    wandb_run_id = state["wandb_run_id"] if state is not None else None
    run = wandb.init(
        project="miccai2024_ssl_debug", config=args.to_dict(), id=wandb_run_id, resume="allow", 
        name=args.name
    )
    wandb_run_id = run.id

    set_global_seed(args.seed)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    ssl_loader, train_loader, all_train_loader, val_loader, test_loader = make_data_loaders(args)

    backbone = resnet10t_instance_norm()
    model = VICReg(backbone, proj_dims=[512, 512, 2048], features_dim=512).to(DEVICE)
    if state is not None:
        model.load_state_dict(state["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])

    cosine_scheduler = make_cosine_scheduler(
        1e-4, 0, epochs=args.epochs, niter_per_ep=len(ssl_loader)
    )

    best_score = 0.0 if state is None else state["best_score"]
    start_epoch = 0 if state is None else state["epoch"]
    early_stopping_counter = 0 if state is None else state['early_stopping_counter']

    if state is not None:
        set_all_rng_states(state["rng_states"])

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch}")

        if args.checkpoint_path is not None:
            print("Saving checkpoint")
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": best_score,
                "epoch": epoch,
                "rng_states": get_all_rng_states(),
                "wandb_run_id": wandb_run_id,
                "early_stopping_counter": early_stopping_counter,
            }
            torch.save(state, args.checkpoint_path)

        print("Running SSL")
        model.train()
        for i, batch in enumerate(tqdm(ssl_loader)):
            # set lr
            iter = epoch * len(ssl_loader) + i
            lr = cosine_scheduler[iter]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            wandb.log({"lr": lr})

            optimizer.zero_grad()
            p1, p2 = batch
            p1, p2 = p1.to(DEVICE), p2.to(DEVICE)
            loss = model(p1, p2)
            wandb.log({"ssl_loss": loss.item()})
            loss.backward()
            optimizer.step()

        metrics, _ = run_linear_probing(model, train_loader, val_loader)
        score = metrics["auc"]
        metrics = {f"val/{k}": v for k, v in metrics.items()}
        metrics['epoch'] = epoch

        wandb.log(metrics)

        if score > best_score:
            early_stopping_counter = 0
            best_score = score
            best_model_state = model.state_dict()

            if args.run_test: 
                _, table = run_linear_probing(model, train_loader, val_loader, [all_train_loader, test_loader])
                table.to_csv(args.outputs_path, index=False)
            
            if args.save_weights_path is not None:
                torch.save(best_model_state, args.save_weights_path)
        else: 
            early_stopping_counter += 1
            if early_stopping_counter > 20: 
                print(f"Early stopping after {epoch} epochs with no improvement")
                break


@torch.no_grad()
def run_linear_probing(model, train_loader, val_loader, other_loaders=[]):
    print("Running linear probing")
    model.eval()
   
    accumulator = DataFrameCollector()

    X_train = []
    y_train = []
    for i, batch in enumerate(tqdm(train_loader)):
        patch = batch.pop("patch").to(DEVICE)
        y = torch.tensor(
            [0 if grade == "Benign" else 1 for grade in batch["grade"]],
            dtype=torch.long,
        )
        accumulator(batch)
        logging.debug(f"{patch.shape=}, {y.shape=}")
        with torch.no_grad():
            features = model.backbone(patch)
        logging.debug(f"{features.shape=}")
        X_train.append(features)
        y_train.append(y)
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train)
    train_table = accumulator.compute()
    accumulator.reset()

    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train.cpu().numpy(), y_train.cpu().numpy())

    y_pred_train = clf.predict_proba(X_train.cpu().numpy())
    train_table.loc[:, "y_pred"] = y_pred_train[:, 1]

    y_pred = []
    for i, batch in enumerate(tqdm(val_loader)):
        patch = batch.pop("patch").to(DEVICE)
        accumulator(batch)
        with torch.no_grad():
            features = model.backbone(patch)
        y_pred.append(clf.predict_proba(features.cpu().numpy()))

    y_pred = np.concatenate(y_pred, axis=0)
    val_table = accumulator.compute()
    accumulator.reset()
    # insert predictions into table
    val_table.loc[:, "y_pred"] = y_pred[:, 1]

    y_pred_core = val_table.groupby("core_id")["y_pred"].mean()
    val_table["label"] = val_table.grade.apply(lambda g: 0 if g == "Benign" else 1)
    y_true_core = val_table.groupby("core_id")["label"].first()
    score = roc_auc_score(y_true_core, y_pred_core)

    high_involvement_table = val_table[(val_table.pct_cancer > 40) | (val_table.grade == "Benign")]
    y_true_high_involvement = high_involvement_table.groupby("core_id")["label"].first()
    y_pred_high_involvement = high_involvement_table.groupby("core_id")["y_pred"].mean()
    score_high_involvement = roc_auc_score(
        y_true_high_involvement, y_pred_high_involvement
    )
    val_table.drop(columns=["label"], inplace=True)

    train_table = train_table.groupby("core_id")['y_pred'].mean().reset_index()
    val_table = val_table.groupby("core_id")['y_pred'].mean().reset_index()
    
    full_table = pd.concat([train_table, val_table], ignore_index=True)

    for loader in other_loaders:
        y_pred_test = []
        for i, batch in enumerate(tqdm(loader)):
            patch = batch.pop("patch").to(DEVICE)
            accumulator(batch)
            with torch.no_grad():
                features = model.backbone(patch)
            y_pred_test.append(clf.predict_proba(features.cpu().numpy()))
        y_pred_test = np.concatenate(y_pred_test, axis=0)

        current_table = accumulator.compute()
        accumulator.reset()
        current_table.loc[:, "y_pred"] = y_pred_test[:, 1]

        current_table = current_table.groupby("core_id")['y_pred'].mean().reset_index()

        full_table = pd.concat([full_table, current_table], ignore_index=True)

    return {
        "auc": score,
        "auc_high_involvement": score_high_involvement,
    }, full_table


def make_data_loaders(args: Args):
    print("Loading data...")

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    
    train_core_ids = splits["train"]
    val_core_ids = splits["val"]
    test_core_ids = splits["test"]
    ssl_train_core_ids = splits["ssl_train"]

    print(f"SSL Train cores: {len(ssl_train_core_ids)}")
    print(f"Train cores: {len(train_core_ids)}")
    print(f"Val cores: {len(val_core_ids)}")
    print(f"Test cores: {len(test_core_ids)}")

    data_hdf5_file = args.paths.data_h5_path
    metadata_csv_file = args.paths.metadata_csv_path

    if args.data_type == "bmode": 
        print("SSL dataset...")
        ssl_dataset = BModePatchesDataset(
            data_hdf5_file, 
            metadata_csv_file,
            ssl_train_core_ids,
            patch_size=(args.patch_size, args.patch_size),
            stride=(args.patch_stride, args.patch_stride),
            needle_mask_threshold=0.6 if not args.full_prostate else -1,
            prostate_mask_threshold=-1 if not args.full_prostate else 0.1,
            transform=SSLTransform(),
        )
        print("Train dataset...")
        train_dataset = BModePatchesDataset(
            data_hdf5_file, 
            metadata_csv_file,
            train_core_ids,
            patch_size=(args.patch_size, args.patch_size),
            stride=(args.patch_stride, args.patch_stride),
            needle_mask_threshold=0.6,
            prostate_mask_threshold=-1,
            transform=Transform(),
        )
        all_train_dataset = BModePatchesDataset(
            data_hdf5_file, 
            metadata_csv_file,
            ssl_train_core_ids, 
            patch_size=(args.patch_size, args.patch_size),
            stride=(args.patch_stride, args.patch_stride),
            needle_mask_threshold=0.6,
            prostate_mask_threshold=-1,
            transform=Transform(),
        )

        print("Val dataset...")
        val_dataset = BModePatchesDataset(
            data_hdf5_file, 
            metadata_csv_file,
            val_core_ids,
            patch_size=(args.patch_size, args.patch_size),
            stride=(args.patch_stride, args.patch_stride),
            needle_mask_threshold=0.6,
            prostate_mask_threshold=-1,
            transform=Transform(),
        )
        print("Test dataset...")
        test_dataset = BModePatchesDataset(
            data_hdf5_file, 
            metadata_csv_file,
            test_core_ids,
            patch_size=(args.patch_size, args.patch_size),
            stride=(args.patch_stride, args.patch_stride),
            needle_mask_threshold=0.6,
            prostate_mask_threshold=-1,
            transform=Transform(),
        )
    else: 
        print("SSL dataset...")
        ssl_dataset = RFPatchesDataset(
            data_hdf5_file, 
            metadata_csv_file,
            ssl_train_core_ids,
            patch_size_mm=args.patch_size_mm,
            patch_stride_mm=args.patch_stride_mm,
            needle_mask_threshold=0.6,
            prostate_mask_threshold=0.1,
            transform=SSLTransform(),
        )
        print("Train dataset...")
        train_dataset = RFPatchesDataset(
            data_hdf5_file, 
            metadata_csv_file,
            train_core_ids,
            patch_size_mm=args.patch_size_mm,
            patch_stride_mm=args.patch_stride_mm,
            needle_mask_threshold=0.6,
            prostate_mask_threshold=0.1,
            transform=Transform(),
        )
        all_train_dataset = RFPatchesDataset(
            data_hdf5_file, 
            metadata_csv_file,
            ssl_train_core_ids, 
            patch_size_mm=args.patch_size_mm,
            patch_stride_mm=args.patch_stride_mm,
            needle_mask_threshold=0.6,
            prostate_mask_threshold=0.1,
            transform=Transform(),
        )
        print("Val dataset...")
        val_dataset = RFPatchesDataset(
            data_hdf5_file, 
            metadata_csv_file,
            val_core_ids,
            patch_size_mm=args.patch_size_mm,
            patch_stride_mm=args.patch_stride_mm,
            needle_mask_threshold=0.6,
            prostate_mask_threshold=0.1,
            transform=Transform(),
        )
        print("Test dataset...")
        test_dataset = RFPatchesDataset(
            data_hdf5_file, 
            metadata_csv_file,
            test_core_ids,
            patch_size_mm=args.patch_size_mm,
            patch_stride_mm=args.patch_stride_mm,
            needle_mask_threshold=0.6,
            prostate_mask_threshold=0.1,
            transform=Transform(),
        )

    ssl_loader = torch.utils.data.DataLoader(
        ssl_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    all_train_loader = torch.utils.data.DataLoader(
        all_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"SSL Train batches: {len(ssl_loader)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return ssl_loader, train_loader, all_train_loader, val_loader, test_loader


if __name__ == "__main__":
    args = parse_args()
    main(args)

import json
import os
from argparse import ArgumentParser, Namespace

import torch
from src.data_factory import DataLoaderFactory
from train_prostnfound import MaskedPredictionModule, ProstNFound

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import LBFGS
from sklearn.metrics import recall_score, roc_auc_score, roc_curve
from src.utils import calculate_metrics
import h5py
from matplotlib.colors import ListedColormap
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import dilation


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        required=True,
        help="""Path to the `.pth` file holding the saved model weights. The config of the model should be saved as `config.json` in the same directory""",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="""Path to the directory where metrics and heatmaps will be saved.""",
    )
    parser.add_argument(
        '--benchmark_only', action='store_true', help="If true, only run the benchmarking code"
    )

    return parser.parse_args()


def main(args):
    model_path = args.model_path
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    config = Namespace(**config)

    # instantiate the dataset with the same config
    print(f"Test center: {config.test_center}")
    print(f"Instantiating dataloaders...")
    assert config.test_center is not None, "Test center must be provided"
    loader_factory = DataLoaderFactory(
        fold=config.fold,
        n_folds=config.n_folds,
        test_center=config.test_center,
        undersample_benign_ratio=config.undersample_benign_ratio,
        min_involvement_train=config.min_involvement_train,
        batch_size=config.batch_size,
        image_size=config.image_size,
        mask_size=config.mask_size,
        augmentations=config.augmentations,
        remove_benign_cores_from_positive_patients=config.remove_benign_cores_from_positive_patients,
        val_seed=config.val_seed,
        limit_train_data=config.limit_train_data
        if config.limit_train_data < 1
        else None,
        train_subset_seed=config.train_subsample_seed,
        rf_as_bmode=config.__dict__.get("rf_as_bmode", False),
        include_rf= "sparse_cnn_patch_features_rf" in config.prompts,
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # instantiate the model with the same config
    model = ProstNFound.from_args(config)
    print(model.load_state_dict(torch.load(model_path, map_location="cpu")))

    # copy the model weights and config to the output directory
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pth"))
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f)

    # extract all pixel predictions from val loader
    pixel_preds, pixel_labels, core_ids = extract_all_pixel_predictions(
        model, loader_factory.val_loader()
    )
    core_ids = np.array(core_ids)

    # fit temperature and bias to center and scale the predictions
    temp = nn.Parameter(torch.ones(1))
    bias = nn.Parameter(torch.zeros(1))

    optim = LBFGS([temp, bias], lr=1e-3, max_iter=100, line_search_fn="strong_wolfe")

    # weight the loss to account for class imbalance
    pos_weight = (1 - pixel_labels).sum() / pixel_labels.sum()
    # encourage sensitivity over specificity
    pos_weight *= 1.6

    def closure():
        optim.zero_grad()
        logits = pixel_preds / temp + bias
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits[:, 0], pixel_labels)
        loss.backward()
        return loss

    for i in range(10):
        print(optim.step(closure))

    pixel_preds_tc = pixel_preds / temp + bias
    val_outputs = get_core_predictions_from_pixel_predictions(
        pixel_preds_tc, pixel_labels, core_ids
    )
    val_outputs.to_csv(os.path.join(args.output_dir, "val_outputs.csv"))

    # extract all pixel predictions from test loader
    pixel_preds, pixel_labels, core_ids = extract_all_pixel_predictions(
        model, loader_factory.test_loader()
    )
    core_ids = np.array(core_ids)
    test_outputs = get_core_predictions_from_pixel_predictions(
        pixel_preds / temp + bias, pixel_labels, core_ids
    )
    test_outputs.to_csv(os.path.join(args.output_dir, "test_outputs.csv"))

    print("Validation Metrics")
    core_preds_val = val_outputs["core_pred"].values
    core_labels_val = val_outputs["core_label"].values
    core_preds_test = test_outputs["core_pred"].values
    core_labels_test = test_outputs["core_label"].values

    print("sens", recall_score(core_labels_val, core_preds_val > 0.5))
    print("spec", recall_score(core_labels_val, core_preds_val > 0.5, pos_label=0))
    print(calculate_metrics(core_preds_val, core_labels_val, log_images=False))

    print("Test Metrics")
    print("sens", recall_score(core_labels_test, core_preds_test > 0.5))
    print("spec", recall_score(core_labels_test, core_preds_test > 0.5, pos_label=0))
    print(calculate_metrics(core_preds_test, core_labels_test, log_images=False))

    # make temperature calibrated model
    tc_layer = nn.Conv2d(1, 1, 1)
    tc_layer.weight.data[0, 0, 0, 0] = temp.data
    tc_layer.bias.data[0] = bias.data

    class TCModel(nn.Module):
        def __init__(self, model, tc_layer):
            super().__init__()
            self.model = model
            self.tc_layer = tc_layer

        def forward(self, x, *args, **kwargs):
            x = self.model(x, *args, **kwargs)
            x = self.tc_layer(x)
            return x

    tc_model = TCModel(model, tc_layer).cuda()

    # ========================================
    # EXPORT heatmap predictions
    # ========================================

    outputs_path = os.path.join(args.output_dir, "heatmaps.h5")
    with h5py.File(outputs_path, "w") as f:
        for batch in tqdm(loader_factory.test_loader(), desc="Exporting heatmaps"):
            (
                heatmap_logits,
                bmode,
                prostate_mask,
                needle_mask,
                core_id,
            ) = extract_heatmap_and_data(tc_model, batch)
            f.create_group(str(core_id))
            f[str(core_id)].create_dataset("heatmap_logits", data=heatmap_logits)
            f[str(core_id)].create_dataset("bmode", data=bmode)
            f[str(core_id)].create_dataset("prostate_mask", data=prostate_mask)
            f[str(core_id)].create_dataset("needle_mask", data=needle_mask)


@torch.no_grad()
def extract_heatmap_and_data(model, batch, device=DEVICE):
    batch = batch.copy()
    bmode = batch.pop("bmode").to(device)
    needle_mask = batch.pop("needle_mask").to(device)
    prostate_mask = batch.pop("prostate_mask").to(device)

    psa = batch["psa"].to(device)
    age = batch["age"].to(device)
    label = batch["label"].to(device)
    family_history = batch["family_history"].to(device)
    anatomical_location = batch["loc"].to(device)
    approx_psa_density = batch["approx_psa_density"].to(device)
    if 'rf' in batch: 
        rf = batch.pop("rf").to(device)
    else: 
        rf = None 

    core_id = batch["core_id"][0]

    B = len(bmode)
    task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

    heatmap_logits = model(
        bmode,
        task_id=task_id,
        anatomical_location=anatomical_location,
        psa=psa,
        age=age,
        family_history=family_history,
        prostate_mask=prostate_mask,
        needle_mask=needle_mask,
        approx_psa_density=approx_psa_density,
        rf_image = rf
    ).cpu()

    heatmap_logits = heatmap_logits[0, 0].sigmoid().cpu().numpy()
    bmode = bmode[0, 0].cpu().numpy()
    prostate_mask = prostate_mask[0, 0].cpu().numpy()
    needle_mask = needle_mask[0, 0].cpu().numpy()
    core_id = core_id

    return heatmap_logits, bmode, prostate_mask, needle_mask, core_id


def extract_all_pixel_predictions(model, loader):
    pixel_labels = []
    pixel_preds = []
    core_ids = []

    model.eval()
    model.to(DEVICE)

    for i, batch in enumerate(tqdm(loader)):
        with torch.no_grad():
            bmode = batch.pop("bmode").to(DEVICE)
            needle_mask = batch.pop("needle_mask").to(DEVICE)
            prostate_mask = batch.pop("prostate_mask").to(DEVICE)

            psa = batch["psa"].to(DEVICE)
            age = batch["age"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            involvement = batch["involvement"].to(DEVICE)
            family_history = batch["family_history"].to(DEVICE)
            anatomical_location = batch["loc"].to(DEVICE)
            approx_psa_density = batch["approx_psa_density"].to(DEVICE)
            if 'rf' in batch: 
                rf = batch.pop("rf").to(DEVICE)
            else: 
                rf = None 

            B = len(bmode)
            task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

            # run the model
            heatmap_logits = model(
                bmode,
                task_id=task_id,
                anatomical_location=anatomical_location,
                psa=psa,
                age=age,
                family_history=family_history,
                prostate_mask=prostate_mask,
                needle_mask=needle_mask,
                approx_psa_density=approx_psa_density,
                rf_image = rf
            )

            # compute predictions
            masks = (prostate_mask > 0.5) & (needle_mask > 0.5)

            predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)

            labels = torch.zeros(len(predictions), device=predictions.device)
            for i in range(len(predictions)):
                labels[i] = label[batch_idx[i]]
            pixel_preds.append(predictions.cpu())
            pixel_labels.append(labels.cpu())

            core_id = batch["core_id"]
            core_ids.extend(core_id[batch_idx[i]] for i in range(len(predictions)))

    pixel_preds = torch.cat(pixel_preds)
    pixel_labels = torch.cat(pixel_labels)

    return pixel_preds, pixel_labels, core_ids


def get_bmode_cmap():
    
    # Load the array
    A = np.load('G7.npy')

    # Normalize to the range 0-1
    A = A / 255.0

    # Create a colormap
    cmap = ListedColormap(A)
    return cmap


def render_heatmap_v1(heatmap, bmode, prostate_mask, needle_mask, metadata): 
    cmap = get_bmode_cmap()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    extent=(0, 46, 28, 0)

    heatmap_logits = np.flip(heatmap.copy(), axis=0)
    bmode = np.flip(bmode, axis=0)
    prostate_mask = np.flip(prostate_mask, axis=0)
    needle_mask = np.flip(needle_mask, axis=0)

    prostate_mask_for_alpha = resize(prostate_mask, (heatmap_logits.shape[0], heatmap_logits.shape[1]), order=0)
    # expand the prostate mask
    prostate_mask_for_alpha = dilation(prostate_mask_for_alpha)
    prostate_mask_for_alpha = gaussian(prostate_mask_for_alpha, sigma=5)
    
    heatmap_logits = gaussian(heatmap_logits, sigma=3)

    ax[1].imshow(bmode, extent=extent, cmap=cmap)
    ax[1].imshow(heatmap_logits, vmin=0, vmax=1, extent=extent, cmap='jet', alpha=prostate_mask_for_alpha*0.5)
    ax[0].imshow(bmode, extent=extent, cmap=cmap)
    ax[0].imshow(prostate_mask, extent=extent, cmap='Purples', alpha=0.5*prostate_mask)
    ax[0].imshow(needle_mask, extent=extent, alpha=needle_mask*0.5)

    fig.suptitle(f'Core {metadata["core_id"]} Grade {metadata["grade"]} Involvement {metadata["pct_cancer"]}')

    for a in ax:
        a.axis('off')

    fig.tight_layout()


def render_heatmap_v2(heatmap, bmode, prostate_mask, needle_mask, metadata): 
    cmap = get_bmode_cmap()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    extent=(0, 46, 28, 0)

    heatmap_logits = np.flip(heatmap.copy(), axis=0)
    bmode = np.flip(bmode, axis=0)
    prostate_mask = np.flip(prostate_mask, axis=0)
    needle_mask = np.flip(needle_mask, axis=0)

    prostate_mask_for_alpha = resize(prostate_mask, (heatmap_logits.shape[0], heatmap_logits.shape[1]), order=0)
    # expand the prostate mask
    prostate_mask_for_alpha = dilation(prostate_mask_for_alpha)
    prostate_mask_for_alpha = gaussian(prostate_mask_for_alpha, sigma=3)
    
    heatmap_logits = gaussian(heatmap_logits, sigma=1)

    ax[1].imshow(bmode, extent=extent, cmap=cmap)
    ax[1].imshow(heatmap_logits, vmin=0, vmax=1, extent=extent, cmap='jet', alpha=prostate_mask_for_alpha*0.5)
    ax[0].imshow(bmode, extent=extent, cmap=cmap)
    ax[0].imshow(prostate_mask, extent=extent, cmap='Purples', alpha=0.5*prostate_mask)
    ax[0].imshow(needle_mask, extent=extent, alpha=needle_mask*0.5)

    fig.suptitle(f'Core {metadata["core_id"]} Grade {metadata["grade"]} Involvement {metadata["pct_cancer"]}')

    for a in ax:
        a.axis('off')

    fig.tight_layout()


def get_core_predictions_from_pixel_predictions(pixel_preds, pixel_labels, core_ids):
    data = []
    for core in np.unique(core_ids):
        mask = core_ids == core
        core_pred = pixel_preds[mask].sigmoid().mean().item()
        core_label = pixel_labels[mask][0].item()
        data.append({"core_id": core, "core_pred": core_pred, "core_label": core_label})

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    main(parse_args())

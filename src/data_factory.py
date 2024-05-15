from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image, Mask
from .transform import RandomTranslation

from .nct2013.data_access import data_accessor
from .nct2013.bmode_dataset import BModeDatasetV1
from .nct2013.cohort_selection import select_cohort
from sklearn.model_selection import StratifiedShuffleSplit

# Here, we normalize the scalar prompt features according to the min/max
table = data_accessor.get_metadata_table()
psa_min = table["psa"].min()
psa_max = table["psa"].max()
psa_avg = table["psa"].mean()
age_min = table["age"].min()
age_max = table["age"].max()
age_avg = table["age"].mean()
approx_psa_density_min = table["approx_psa_density"].min()
approx_psa_density_max = table["approx_psa_density"].max()
approx_psa_density_avg = table["approx_psa_density"].mean()

# Sextant core locations are converted to integer codes -
# here we assume left/right prostate side should have the
# same code
CORE_LOCATION_TO_IDX = {
    "LML": 0,
    "RBL": 1,
    "LMM": 2,
    "RMM": 2,
    "LBL": 1,
    "LAM": 3,
    "RAM": 3,
    "RML": 0,
    "LBM": 4,
    "RAL": 5,
    "RBM": 4,
    "LAL": 5,
}


class TransformV2:
    def __init__(
        self,
        augment="translate",
        image_size=1024,
        mask_size=256,
        dataset_name="nct",
        labeled=True,
    ):
        self.augment = augment
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.labeled = labeled
        self.mask_size = mask_size

    def __call__(self, item):
        out = item.copy()
        bmode = item["bmode"]
        bmode = torch.from_numpy(bmode.copy()).float()
        bmode = bmode.unsqueeze(0)
        bmode = T.Resize((self.image_size, self.image_size), antialias=True)(bmode)
        bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
        bmode = bmode.repeat(3, 1, 1)
        bmode = Image(bmode)
        if not self.labeled:
            return {"bmode": bmode}

        needle_mask = item["needle_mask"]
        needle_mask = needle_mask = torch.from_numpy(needle_mask.copy()).float()
        needle_mask = needle_mask.unsqueeze(0)
        needle_mask = T.Resize(
            (self.image_size, self.image_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(needle_mask)
        needle_mask = Mask(needle_mask)

        prostate_mask = item["prostate_mask"]
        prostate_mask = prostate_mask = torch.from_numpy(prostate_mask.copy()).float()
        prostate_mask = prostate_mask.unsqueeze(0)
        prostate_mask = T.Resize(
            (self.image_size, self.image_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(prostate_mask)
        prostate_mask = Mask(prostate_mask)

        if item.get("rf") is not None:
            rf = item["rf"]
            rf = torch.from_numpy(rf.copy()).float()
            rf = rf.unsqueeze(0)
            if rf.shape != (2504, 512):
                rf = T.Resize((2504, 512), antialias=True)(rf)
            rf = rf.repeat(3, 1, 1)

            if self.augment == "translate":

                bmode, rf, needle_mask, prostate_mask = RandomTranslation(
                    translation=(0.2, 0.2)
                )(bmode, rf, needle_mask, prostate_mask)

        else:
            bmode, needle_mask, prostate_mask = RandomTranslation(
                translation=(0.2, 0.2)
            )(bmode, needle_mask, prostate_mask)

        # interpolate the masks to the mask size
        needle_mask = T.Resize(
            (self.mask_size, self.mask_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(needle_mask)
        prostate_mask = T.Resize(
            (self.mask_size, self.mask_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(prostate_mask)

        out["bmode"] = bmode
        out["needle_mask"] = needle_mask
        out["prostate_mask"] = prostate_mask

        if item.get("rf") is not None:
            out["rf"] = rf

        out["label"] = torch.tensor(item["grade"] != "Benign").long()
        pct_cancer = item["pct_cancer"]
        if np.isnan(pct_cancer):
            pct_cancer = 0
        out["involvement"] = torch.tensor(pct_cancer / 100).float()

        psa = item["psa"]
        if np.isnan(psa):
            psa = psa_avg
        psa = (psa - psa_min) / (psa_max - psa_min)
        out["psa"] = torch.tensor([psa]).float()

        age = item["age"]
        if np.isnan(age):
            age = age_avg
        age = (age - age_min) / (age_max - age_min)
        out["age"] = torch.tensor([age]).float()

        approx_psa_density = item["approx_psa_density"]
        if np.isnan(approx_psa_density):
            approx_psa_density = approx_psa_density_avg
        approx_psa_density = (approx_psa_density - approx_psa_density_min) / (
            approx_psa_density_max - approx_psa_density_min
        )
        out["approx_psa_density"] = torch.tensor([approx_psa_density]).float()

        if item["family_history"] is True:
            out["family_history"] = torch.tensor(1).long()
        elif item["family_history"] is False:
            out["family_history"] = torch.tensor(0).long()
        elif np.isnan(item["family_history"]):
            out["family_history"] = torch.tensor(2).long()

        out["center"] = item["center"]
        loc = item["loc"]
        out["loc"] = torch.tensor(CORE_LOCATION_TO_IDX[loc]).long()
        out["all_cores_benign"] = torch.tensor(item["all_cores_benign"]).bool()
        out["dataset_name"] = self.dataset_name
        out["primary_grade"] = item["primary_grade"]
        out["secondary_grade"] = item["secondary_grade"]
        out["grade"] = item["grade"]
        out["core_id"] = item["core_id"]

        return out


class DataLoaderFactory:
    def __init__(
        self,
        fold: int = 0,
        n_folds: int = 5,
        test_center: str = None,
        undersample_benign_ratio: float = 3.0,
        min_involvement_train: float = 40,
        remove_benign_cores_from_positive_patients: bool = True,
        batch_size: int = 1,
        image_size: int = 1024,
        mask_size: int = 256,
        augmentations: str = "none",
        labeled: bool = True,
        limit_train_data: float | None = None,
        train_subset_seed: int = 0,
        val_seed: int = 0,
        rf_as_bmode: bool = False,
        include_rf: bool = True,
    ):

        train_cores, val_cores, test_cores = select_cohort(
            fold=fold,
            n_folds=n_folds,
            test_center=test_center,
            undersample_benign_ratio=undersample_benign_ratio,
            involvement_threshold_pct=min_involvement_train,
            exclude_benign_cores_from_positive_patients=remove_benign_cores_from_positive_patients,
            splits_file="/ssd005/projects/exactvu_pca/nct2013/patient_splits.csv",
            val_seed=val_seed,
        )

        if limit_train_data is not None:
            cores = train_cores
            center = [core.split("-")[0] for core in cores]

            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1 - limit_train_data,
                random_state=train_subset_seed,
            )
            for train_index, _ in sss.split(cores, center):
                train_cores = [cores[i] for i in train_index]

        self.train_transform = TransformV2(
            augment=augmentations,
            image_size=image_size,
            labeled=labeled,
            mask_size=mask_size,
        )
        self.val_transform = TransformV2(
            augment="none", image_size=image_size, labeled=labeled, mask_size=mask_size
        )

        self.train_dataset = BModeDatasetV1(
            train_cores,
            self.train_transform,
            rf_as_bmode=rf_as_bmode,
            include_rf=include_rf,
        )
        self.val_dataset = BModeDatasetV1(
            val_cores,
            self.val_transform,
            rf_as_bmode=rf_as_bmode,
            include_rf=include_rf,
        )
        self.test_dataset = BModeDatasetV1(
            test_cores,
            self.val_transform,
            rf_as_bmode=rf_as_bmode,
            include_rf=include_rf,
        )

        self.batch_size = batch_size
        self.labeled = labeled

    def train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

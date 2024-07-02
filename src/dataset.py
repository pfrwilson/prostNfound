from simple_parsing.helpers.serialization.serializable import FrozenSerializable
from torch.utils.data import Dataset
import os
import json
import h5py
import pandas as pd
import numpy as np

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image as ImageTV, Mask as MaskTV
from .transform import RandomTranslation
from .utils import PatchView

import json
import os

import h5py
import numpy as np
import pandas as pd
from skimage.transform import resize
from PIL import Image
from tqdm import tqdm
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
import logging
from dataclasses import dataclass



class DataAccessor:
    def __init__(self, data_h5_path, metadata_csv_path):
        self.data_h5_path = data_h5_path
        self.metadata = pd.read_csv(metadata_csv_path, index_col=0)
        self.metadata = self.metadata.sort_values(by=["core_id"]).reset_index(drop=True)

    def get_metadata_table(self):
        return self.metadata

    def get_rf_image(self, core_id):
        with h5py.File(self.data_h5_path, "r") as f:
            arr = f["rf"][core_id][..., 0]
        return arr

    def get_bmode_image(self, core_id):
        with h5py.File(self.data_h5_path, "r") as f:
            arr = f["bmode"][core_id][..., 0]
        return arr

    def get_prostate_mask(self, core_id):
        with h5py.File(self.data_h5_path, "r") as f:
            return f["prostate_mask"][core_id][:]

    def get_needle_mask(self, core_id):
        with h5py.File(self.data_h5_path, "r") as f:
            return f["needle_mask"][core_id][:]

    def get_metadata(self, core_id):
        return (
            self.get_metadata_table()
            .loc[self.get_metadata_table()["core_id"] == core_id]
            .iloc[0]
            .to_dict()
        )

    def load_or_create_resized_bmode_data(self, image_size):
        PREPROCESSED_DATA_DIR = os.path.join(
            os.path.dirname(self.data_h5_path), "preprocessed_data"
        )

        dataset_dir = os.path.join(
            PREPROCESSED_DATA_DIR, f"images_{image_size[0]}x{image_size[1]}"
        )
        if not os.path.exists(dataset_dir):
            print(f"Creating preprocessed dataset at {dataset_dir}")

            core_ids = sorted(self.get_metadata_table().core_id.unique().tolist())
            data_buffer = np.zeros((len(core_ids), *image_size), dtype=np.uint8)
            for i, core_id in enumerate(
                tqdm(core_ids, desc="Preprocessing B-mode images")
            ):
                bmode = self.get_bmode_image(core_id)
                bmode = bmode / 255.0
                from skimage.transform import resize

                bmode = resize(bmode, image_size, anti_aliasing=True)
                bmode = (bmode * 255).astype(np.uint8)
                data_buffer[i, ...] = bmode

            print(f"Saving preprocessed dataset at {dataset_dir}")
            os.makedirs(dataset_dir, exist_ok=True)
            np.save(os.path.join(dataset_dir, "bmode.npy"), data_buffer)
            core_id_2_idx = {core_id: idx for idx, core_id in enumerate(core_ids)}
            with open(os.path.join(dataset_dir, "core_id_2_idx.json"), "w") as f:
                json.dump(core_id_2_idx, f)

        bmode_data = np.load(os.path.join(dataset_dir, "bmode.npy"), mmap_mode="r")
        with open(os.path.join(dataset_dir, "core_id_2_idx.json"), "r") as f:
            core_id_2_idx = json.load(f)

        return bmode_data, core_id_2_idx



@dataclass(frozen=True)
class CohortSelectionOptions(FrozenSerializable): 
    """Config for cohort selection. Basic strategies for train/test selection are (center-balanced) k-fold
    or leave-one-center-out (LOCO) for test set. In either case the train set is further subdivided into 
    a center-balanced train and validation set. The train set can be further filtered using provided options.
    Cohort selection is done by splitting the patients into train/val/test sets, and then selecting the cores
    for the given split. 
    
    Args: 
        fold: If specified, the fold to use for the train/val/test split.
        n_folds (int): If specified, the number of folds to use for the train/val/test split.
        test_center (str): If specified, the center to use for the test set.
        val_seed (int): seed for validation split
        val_train_ratio (float): ratio validation to training
        undersample_benign_ratio (float): ratio to undersample benign cores (Train only)
        min_involvement_train (float): minimum involvement for training (Train only - use a percentage.)
        remove_benign_cores_from_positive_patients (bool): remove benign cores from positive patients (Train only)
        limit_train_data (float): If not none, subsample the training data to the given ratio of total training data
        train_subsample_seed (int): seed for subsampling the training data
    """
    fold: int | None = None
    n_folds: int | None = None 
    test_center: str | None = None 
    val_seed: int = 0 
    val_train_ratio: float = 0.2 
    undersample_benign_ratio: float | None = 6 
    min_involvement_train: float = 40
    remove_benign_cores_from_positive_patients: bool = True
    limit_train_data: float | None = None 
    train_subsample_seed: int | None = 42


class CohortSelector:
    def __init__(self, metadata_table):
        self.metadata_table = metadata_table

    def get_patient_splits_by_fold(self, fold=0, n_folds=5, val_size=0.2, val_seed=0):
        """returns the list of patient ids for the train, val, and test splits."""

        if fold not in range(n_folds):
            raise ValueError(f"Fold must be in range {n_folds}, but got {fold}")

        metadata_table = self.metadata_table
        patient_table = metadata_table.drop_duplicates(subset=["patient_id"])
        patient_table = patient_table[["patient_id", "center"]]

        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        for i, (train_idx, test_idx) in enumerate(
            kfold.split(patient_table["patient_id"], patient_table["center"])
        ):
            if i == fold:
                train = patient_table.iloc[train_idx]
                test = patient_table.iloc[test_idx]
                break

        train, val = train_test_split(
            train, test_size=val_size, random_state=val_seed, stratify=train["center"]
        )

        train = train.patient_id.values.tolist()
        val = val.patient_id.values.tolist()
        test = test.patient_id.values.tolist()

        return train, val, test

    def get_patient_splits_by_center(self, leave_out="UVA", val_size=0.2, val_seed=0):
        """returns the list of patient ids for the train, val, and test splits."""
        if leave_out not in self.metadata_table['center'].unique():
            raise ValueError(
                f"leave_out must be one of 'UVA', 'CRCEO', 'PCC', 'PMCC', 'JH', but got {leave_out}"
            )

        metadata_table = self.metadata_table
        patient_table = metadata_table.drop_duplicates(subset=["patient_id"])
        table = patient_table[["patient_id", "center"]]

        train = table[table.center != leave_out]
        train, val = train_test_split(
            train, test_size=val_size, random_state=val_seed, stratify=train["center"]
        )
        train = train.patient_id.values.tolist()
        val = val.patient_id.values.tolist()
        test = table[table.center == leave_out].patient_id.values.tolist()

        return train, val, test

    def get_core_ids(self, patient_ids):
        """returns the list of core ids for the given patient ids."""
        metadata_table = self.metadata_table
        return metadata_table[
            metadata_table.patient_id.isin(patient_ids)
        ].core_id.values.tolist()

    def remove_benign_cores_from_positive_patients(self, core_ids):
        """Returns the list of cores in the given list that are either malignant or from patients with no malignant cores."""
        table = self.metadata_table.copy()
        table["positive"] = table.grade.apply(lambda g: 0 if g == "Benign" else 1)
        num_positive_for_patient = table.groupby("patient_id").positive.sum()
        num_positive_for_patient.name = "patients_positive"
        table = table.join(num_positive_for_patient, on="patient_id")
        ALLOWED = table.query(
            "positive == 1 or patients_positive == 0"
        ).core_id.to_list()

        return [core for core in core_ids if core in ALLOWED]

    def remove_cores_below_threshold_involvement(self, core_ids, threshold_pct):
        """Returns the list of cores with at least the given percentage of cancer cells."""
        table = self.metadata_table.copy()
        ALLOWED = table.query(
            "grade == 'Benign' or pct_cancer >= @threshold_pct"
        ).core_id.to_list()
        return [core for core in core_ids if core in ALLOWED]

    def undersample_benign(self, cores, seed=0, benign_to_cancer_ratio=1):
        """Returns the list of cores with the same cancer cores and the benign cores undersampled to the given ratio."""

        table = self.metadata_table.copy()
        benign = table.query('grade == "Benign"').core_id.to_list()
        cancer = table.query('grade != "Benign"').core_id.to_list()
        import random

        cores_benign = [core for core in cores if core in benign]
        cores_cancer = [core for core in cores if core in cancer]
        rng = random.Random(seed)
        cores_benign = rng.sample(
            cores_benign, int(len(cores_cancer) * benign_to_cancer_ratio)
        )

        return [core for core in cores if core in cores_benign or core in cores_cancer]

    def apply_core_filters(
        self,
        core_ids,
        exclude_benign_cores_from_positive_patients=False,
        involvement_threshold_pct=None,
        undersample_benign_ratio=None,
    ):
        if exclude_benign_cores_from_positive_patients:
            core_ids = self.remove_benign_cores_from_positive_patients(core_ids)

        if involvement_threshold_pct is not None:
            if involvement_threshold_pct < 0 or involvement_threshold_pct > 100:
                raise ValueError(
                    f"involvement_threshold_pct must be between 0 and 100, but got {involvement_threshold_pct}"
                )
            core_ids = self.remove_cores_below_threshold_involvement(
                core_ids, involvement_threshold_pct
            )

        if undersample_benign_ratio is not None:
            core_ids = self.undersample_benign(
                core_ids, seed=0, benign_to_cancer_ratio=undersample_benign_ratio
            )

        return core_ids

    def select_cohort(
        self,
        fold=None,
        n_folds=None,
        test_center=None,
        exclude_benign_cores_from_positive_patients=False,
        involvement_threshold_pct=None,
        undersample_benign_ratio=None,
        seed=0,
        splits_file=None,
        val_seed=0,
        val_size=0.2,
    ):
        """Returns the list of core ids for the given cohort selection criteria.

        Default is to use the 5-fold split.

        Args:
            fold (int): If specified, the fold to use for the train/val/test split.
            n_folds (int): If specified, the number of folds to use for the train/val/test split.
            test_center (str): If specified, the center to use for the test set.

            The following arguments are used to filter the cores in the cohort, affecting
                ONLY THE TRAIN SET:
            remove_benign_cores_from_positive_patients (bool): If True, remove cores from patients with malignant cores that also have benign cores.
                Only applies to the training set.
            involvement_threshold_pct (float): If specified, remove cores with less than the given percentage of cancer cells.
                this should be a value between 0 and 100. Only applies to the training set.
            undersample_benign_ratio (float): If specified, undersample the benign cores to the given ratio. Only applies to the training set.
            seed (int): Random seed to use for the undersampling.
            splits_file: if specified, use the given csv file to load the train/val/test splits (kfold only)
        """

        if test_center is not None:
            logging.info(f"Using test center {test_center}")
            train, val, test = get_patient_splits_by_center(
                leave_out=test_center, val_size=val_size, val_seed=val_seed
            )
        elif fold is not None:
            assert n_folds is not None, "Must specify n_folds if fold is specified."
            train, val, test = get_patient_splits_by_fold(
                fold=fold, n_folds=n_folds, splits_file=splits_file
            )
        else:
            logging.info("Using default 5-fold split.")
            train, val, test = get_patient_splits_by_fold(
                fold=0, n_folds=5, splits_file=splits_file
            )

        train_cores = get_core_ids(train)
        val_cores = get_core_ids(val)
        test_cores = get_core_ids(test)

        train_cores = apply_core_filters(
            train_cores,
            exclude_benign_cores_from_positive_patients=exclude_benign_cores_from_positive_patients,
            involvement_threshold_pct=involvement_threshold_pct,
            undersample_benign_ratio=undersample_benign_ratio,
        )

        # if exclude_benign_cores_from_positive_patients:
        #     train_cores = remove_benign_cores_from_positive_patients(train_cores)
        #
        # if involvement_threshold_pct is not None:
        #     if involvement_threshold_pct < 0 or involvement_threshold_pct > 100:
        #         raise ValueError(
        #             f"involvement_threshold_pct must be between 0 and 100, but got {involvement_threshold_pct}"
        #         )
        #     train_cores = remove_cores_below_threshold_involvement(
        #         train_cores, involvement_threshold_pct
        #     )
        #
        # if undersample_benign_ratio is not None:
        #     train_cores = undersample_benign(
        #         train_cores, seed=seed, benign_to_cancer_ratio=undersample_benign_ratio
        #     )

        return train_cores, val_cores, test_cores

    def select_cohort_from_options(self, options: CohortSelectionOptions): 
        if options.fold is not None:
            train_patients, val_patients, test_patients = (
                self.get_patient_splits_by_fold(
                    fold=options.fold,
                    n_folds=options.n_folds,
                    val_seed=options.val_seed,
                    val_size=options.val_train_ratio,
                )
            )
        else:
            assert (
                options.test_center is not None
            ), "Either fold or test_center must be specified"
            train_patients, val_patients, test_patients = (
                self.get_patient_splits_by_center(
                    leave_out=options.test_center,
                    val_size=options.val_train_ratio,
                    val_seed=options.val_seed,
                )
            )

        train_cores = self.get_core_ids(train_patients)
        val_cores = self.get_core_ids(val_patients)
        test_cores = self.get_core_ids(test_patients)
        ssl_train_cores = train_cores.copy()  # don't filter cores for SSL

        train_cores = self.apply_core_filters(
            train_cores,
            exclude_benign_cores_from_positive_patients=options.remove_benign_cores_from_positive_patients,
            involvement_threshold_pct=options.min_involvement_train,
            undersample_benign_ratio=options.undersample_benign_ratio,
        )

        if options.limit_train_data is not None:
            assert (
                options.limit_train_data <= 1
            ), "limit_train_data should be less than or equal to 1 or None"

            cores = train_cores
            center = [core.split("-")[0] for core in cores]

            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1 - options.limit_train_data,
                random_state=options.train_subsample_seed,
            )
            for train_index, _ in sss.split(cores, center):
                train_cores = [cores[i] for i in train_index]

        return train_cores, val_cores, test_cores, ssl_train_cores


class ProstNFoundDataset(Dataset):
    REQUIRED_METADATA_KEYS = [
        "grade",  # for binary cancer classification
        # "pct_cancer",  # for cancer involvement - used for metric calculations
        "patient_id",
        # "clinically_significant",  # for patient-level classification
    ]

    def __init__(
        self,
        hdf5_file,
        label_csv_file,
        splits_json_file,
        prompts_csv_file=None,
        split="train",
        transform=None,
        include_rf=False,
        rf_as_bmode=False,
    ):
        self.path_to_hdf5_file = hdf5_file
        self.label_csv_file = label_csv_file
        self.prompts_csv_file = prompts_csv_file
        self.transform = transform
        self.split = split
        self.splits_file = splits_json_file
        self.include_rf = include_rf
        self.rf_as_bmode = rf_as_bmode

        assert os.path.exists(
            self.path_to_hdf5_file
        ), f"File {self.path_to_hdf5_file} does not exist"
        assert os.path.exists(
            self.label_csv_file
        ), f"File {self.label_csv_file} does not exist"
        assert os.path.exists(
            self.splits_file
        ), f"File {self.splits_file} does not exist"
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be one of ['train', 'val', 'test']"

        self.data_accessor = DataAccessor(self.path_to_hdf5_file, self.label_csv_file)

        with open(self.splits_file, "r") as f:
            splits = json.load(f)
            self.core_ids = splits[split]

        self.labels_df = pd.read_csv(self.label_csv_file)
        if self.prompts_csv_file is not None:
            self.prompts_df = pd.read_csv(self.prompts_csv_file)
        else:
            self.prompts_df = None

    def __len__(self):
        return len(self.core_ids)

    def __getitem__(self, idx):
        out = {}
        core_id = self.core_ids[idx]

        out["bmode"] = self.get_bmode(core_id)
        out["prostate_mask"] = self.get_prostate_mask(core_id)
        out["needle_mask"] = self.get_needle_mask(core_id)

        if self.include_rf:
            out["rf"] = self.get_rf(core_id)

        labels = self.get_labels(core_id)
        for k, v in labels.items():
            out[k] = v
        prompts = self.get_prompts(core_id)

        for k, v in prompts.items():
            out[k] = v

        if self.transform:
            out = self.transform(out)

        return out

    def get_bmode(self, core_id):
        if self.rf_as_bmode:
            return self.get_rf(core_id)
        return self.data_accessor.get_bmode_image(core_id)

    def get_rf(self, core_id):
        return self.data_accessor.get_rf_image(core_id)

    def get_needle_mask(self, core_id):
        return self.data_accessor.get_needle_mask(core_id)

    def get_prostate_mask(self, core_id):
        return self.data_accessor.get_prostate_mask(core_id)

    def get_labels(self, core_id):
        row = self.data_accessor.get_metadata(core_id)
        for key in self.REQUIRED_METADATA_KEYS:
            if key not in row.keys():
                raise ValueError(
                    f"Key {key} not found in metadata, but required. See {__file__}"
                )

        out = row.copy()

        out["label"] = torch.tensor(out["grade"] != "Benign").long()

        if 'pct_cancer' in out:
            pct_cancer = out["pct_cancer"]
            if np.isnan(pct_cancer):
                pct_cancer = 0
            out["involvement"] = torch.tensor(pct_cancer / 100).float()

        if 'all_cores_benign' in out: 
            out["all_cores_benign"] = torch.tensor(out["all_cores_benign"]).bool()

        return out

    def get_prompts(self, core_id):
        if self.prompts_df is None:
            return {}
        row = (
            self.prompts_df.loc[self.prompts_df["core_id"] == core_id].iloc[0].to_dict()
        )
        row.pop("core_id")
        row = {k: self._convert_prompt(v) for k, v in row.items()}
        return row

    def _convert_prompt(self, prompt):
        if isinstance(prompt, float):
            return torch.tensor([prompt]).float()
        elif isinstance(prompt, int):
            return torch.tensor([prompt]).long()
        else:
            raise ValueError(f"Prompt type {type(prompt)} not supported")


class Transform:
    """
    Handles applying transformations to the image portion of the data.
    """

    def __init__(
        self,
        augment="translate",
        image_size=1024,
        mask_size=256,
    ):
        self.augment = augment
        self.image_size = image_size
        self.mask_size = mask_size

    def __call__(self, item):

        # ======== image data ==========
        out = item.copy()
        bmode = item["bmode"]
        bmode = torch.from_numpy(bmode.copy()).float()
        bmode = bmode.unsqueeze(0)
        bmode = T.Resize((self.image_size, self.image_size), antialias=True)(bmode)
        bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
        bmode = bmode.repeat(3, 1, 1)
        bmode = ImageTV(bmode)

        needle_mask = item["needle_mask"]
        needle_mask = needle_mask = torch.from_numpy(needle_mask.copy()).float()
        needle_mask = needle_mask.unsqueeze(0)
        needle_mask = T.Resize(
            (self.image_size, self.image_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(needle_mask)
        needle_mask = MaskTV(needle_mask)

        prostate_mask = item["prostate_mask"]
        prostate_mask = prostate_mask = torch.from_numpy(prostate_mask.copy()).float()
        prostate_mask = prostate_mask.unsqueeze(0)
        prostate_mask = T.Resize(
            (self.image_size, self.image_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(prostate_mask)
        prostate_mask = MaskTV(prostate_mask)

        if item.get("rf") is not None:
            rf = item["rf"]
            rf = torch.from_numpy(rf.copy()).float()
            rf = rf.unsqueeze(0)
            if rf.shape != (2504, 512):
                rf = T.Resize((2504, 512), antialias=True)(rf)
            rf = rf.repeat(3, 1, 1)

        if self.augment == "translate":
            if item.get("rf") is not None:
                bmode, rf, needle_mask, prostate_mask = RandomTranslation(
                    translation=(0.2, 0.2)
                )(bmode, rf, needle_mask, prostate_mask)
            else:
                bmode, needle_mask, prostate_mask = RandomTranslation(
                    translation=(0.2, 0.2)
                )(bmode, needle_mask, prostate_mask)

        # interpolate the masks back to the mask size
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

        return out


def get_dataloaders_main(
    hdf5_file,
    label_csv_file,
    splits_json_file,
    prompts_csv_file=None,
    augment="translate",
    image_size=1024,
    mask_size=256,
    include_rf=True,
    rf_as_bmode=False,
    batch_size=4,
    num_workers=4,
):
    train_transform = Transform(
        augment=augment, image_size=image_size, mask_size=mask_size
    )
    val_transform = Transform(augment=None, image_size=image_size, mask_size=mask_size)

    train_ds = ProstNFoundDataset(
        hdf5_file,
        label_csv_file,
        splits_json_file,
        prompts_csv_file,
        split="train",
        transform=train_transform,
        include_rf=include_rf,
        rf_as_bmode=rf_as_bmode,
    )
    val_ds = ProstNFoundDataset(
        hdf5_file,
        label_csv_file,
        splits_json_file,
        prompts_csv_file,
        split="val",
        transform=val_transform,
        include_rf=include_rf,
        rf_as_bmode=rf_as_bmode,
    )
    test_ds = ProstNFoundDataset(
        hdf5_file,
        label_csv_file,
        splits_json_file,
        prompts_csv_file,
        split="test",
        transform=val_transform,
        include_rf=include_rf,
        rf_as_bmode=rf_as_bmode,
    )

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_dl = DataLoader(train_ds, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, shuffle=False, **kw)

    return train_dl, val_dl, test_dl


class BModePatchesDataset(Dataset):
    def __init__(
        self,
        hdf5_file,
        label_csv_file,
        core_ids,
        patch_size,
        stride,
        needle_mask_threshold,
        prostate_mask_threshold,
        transform=None,
    ):
        data_accessor = DataAccessor(hdf5_file, label_csv_file)
        self._bmode_data, self._core_id_2_idx = (
            data_accessor.load_or_create_resized_bmode_data((1024, 1024))
        )
        self._metadata_table = data_accessor.get_metadata_table()

        self.core_ids = core_ids

        N = len(self.core_ids)
        self._images = [
            self._bmode_data[self._core_id_2_idx[core_id]] for core_id in core_ids
        ]
        self._prostate_masks = np.zeros((N, 256, 256))
        for i, core_id in enumerate(core_ids):
            self._prostate_masks[i] = data_accessor.get_prostate_mask(core_id)
        self._needle_masks = np.zeros((N, 512, 512))
        for i, core_id in enumerate(core_ids):
            self._needle_masks[i] = data_accessor.get_needle_mask(core_id)
        self._patch_views = PatchView.build_collection_from_images_and_masks(
            self._images,
            window_size=patch_size,
            stride=stride,
            align_to="topright",
            mask_lists=[self._prostate_masks, self._needle_masks],
            thresholds=[prostate_mask_threshold, needle_mask_threshold],
        )
        self._metadata_dicts = []
        for core_id in self.core_ids:
            metadata = (
                self._metadata_table[self._metadata_table.core_id == core_id]
                .iloc[0]
                .to_dict()
            )
            self._metadata_dicts.append(metadata)
        self._indices = []
        for i, pv in enumerate(self._patch_views):
            self._indices.extend([(i, j) for j in range(len(pv))])
        self.transform = transform

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i, j = self._indices[idx]
        pv = self._patch_views[i]
        item = {}
        item["patch"] = pv[j] / 255.0
        metadata = self._metadata_dicts[i].copy()
        item.update(metadata)
        if self.transform is not None:
            item = self.transform(item)
        return item


class RFPatchesDataset(Dataset):

    def __init__(
        self,
        hdf5_file,
        label_csv_file,
        core_ids,
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        needle_mask_threshold=0.6,
        prostate_mask_threshold=-1,
        transform=None,
    ):
        self.data_accessor = DataAccessor(hdf5_file, label_csv_file)
        self._metadata_table = self.data_accessor.get_metadata_table()

        self.core_ids = core_ids
        im_size_mm = 28, 46.06
        im_size_px = self.data_accessor.get_rf_image(core_ids[0]).shape
        self.patch_size_px = int(patch_size_mm[0] * im_size_px[0] / im_size_mm[0]), int(
            patch_size_mm[1] * im_size_px[1] / im_size_mm[1]
        )
        self.patch_stride_px = int(
            patch_stride_mm[0] * im_size_px[0] / im_size_mm[0]
        ), int(patch_stride_mm[1] * im_size_px[1] / im_size_mm[1])

        self._images = []
        for core_id in core_ids:
            image = self.data_accessor.get_rf_image(core_id)
            if image.shape != im_size_px:
                image = resize(image, im_size_px)
            self._images.append(image)

        self._prostate_masks = [
            self.data_accessor.get_prostate_mask(core_id) for core_id in core_ids
        ]
        self._needle_masks = [
            self.data_accessor.get_needle_mask(core_id) for core_id in core_ids
        ]

        self._patch_views = PatchView.build_collection_from_images_and_masks(
            self._images,
            window_size=self.patch_size_px,
            stride=self.patch_stride_px,
            align_to="topright",
            mask_lists=[self._prostate_masks, self._needle_masks],
            thresholds=[prostate_mask_threshold, needle_mask_threshold],
        )
        self._indices = []
        for i, pv in enumerate(self._patch_views):
            self._indices.extend([(i, j) for j in range(len(pv))])

        self.transform = transform

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i, j = self._indices[idx]
        metadata = (
            self._metadata_table[self._metadata_table.core_id == self.core_ids[i]]
            .iloc[0]
            .to_dict()
        )
        pv = self._patch_views[i]
        patch = pv[j]

        patch = patch.copy()
        resize(patch, (256, 256))
        postition = pv.positions[j]

        data = {"patch": patch, **metadata, "position": postition}
        if self.transform is not None:
            data = self.transform(data)

        return data


class PatchesSSLTransform:
    def __call__(self, item):
        patch = item["patch"]
        patch = torch.from_numpy(patch).float()
        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0)

        augs = [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        ]
        p1 = T.Compose(augs)(patch)
        p2 = T.Compose(augs)(patch)

        return p1, p2


class PatchesTransform:
    def __call__(self, item):
        patch = item["patch"]
        patch = torch.from_numpy(patch).float()
        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0)
        item["patch"] = patch
        return item

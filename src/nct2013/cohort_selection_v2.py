import logging
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


class CohortSelector:
    def __init__(self, metadata_table):
        self.metadata_table = metadata_table

    def get_patient_splits_by_fold(self, fold=0, n_folds=5):
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
            train, test_size=0.2, random_state=0, stratify=train["center"]
        )

        train = train.patient_id.values.tolist()
        val = val.patient_id.values.tolist()
        test = test.patient_id.values.tolist()

        return train, val, test

    def get_patient_splits_by_center(self, leave_out="UVA", val_size=0.2, val_seed=0):
        """returns the list of patient ids for the train, val, and test splits."""
        if leave_out not in ["UVA", "CRCEO", "PCC", "PMCC", "JH"]:
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
            train, val, test = self.get_patient_splits_by_center(
                leave_out=test_center, val_size=val_size, val_seed=val_seed
            )
        elif fold is not None:
            assert n_folds is not None, "Must specify n_folds if fold is specified."
            train, val, test = self.get_patient_splits_by_fold(
                fold=fold, n_folds=n_folds, splits_file=splits_file
            )
        else:
            logging.info("Using default 5-fold split.")
            train, val, test = self.get_patient_splits_by_fold(
                fold=0, n_folds=5, splits_file=splits_file
            )

        train_cores = self.get_core_ids(train)
        val_cores = self.get_core_ids(val)
        test_cores = self.get_core_ids(test)

        train_cores = self.apply_core_filters(
            train_cores,
            exclude_benign_cores_from_positive_patients=exclude_benign_cores_from_positive_patients,
            involvement_threshold_pct=involvement_threshold_pct,
            undersample_benign_ratio=undersample_benign_ratio,
        )

        return train_cores, val_cores, test_cores

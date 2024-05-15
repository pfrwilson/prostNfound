from dataclasses import dataclass
import argparse
import json
from rich_argparse import ArgumentDefaultsRichHelpFormatter
import pandas as pd
from src.nct2013.cohort_selection_v2 import CohortSelector
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate splits for the NCT dataset", formatter_class=ArgumentDefaultsRichHelpFormatter)
    parser.add_argument('--data_csv_path', type=str, required=True, help="Path to the data csv file")
    parser.add_argument('--output_filename', type=str, required=True, help="Output filename (.json) where the splits will be saved")

    group = parser.add_argument_group("COHORT SELECTION", 
        description="""Arguments for cohort selection - two basic strategies for test selection are leave-one-center-out and k-fold. 
        If you specify a fold, it will use k-fold cross-validation. If you specify a test center, it will use leave-one-center-out cross-validation.
        In both cases, a validation set is created from the training set.""")
    
    group.add_argument("--fold", type=int, default=None, help="The fold to use. If not specified, uses leave-one-center-out cross-validation.")
    group.add_argument("--n_folds", type=int, default=None, help="The number of folds to use for cross-validation. If not specified, uses leave-one-center-out cross-validation.")
    group.add_argument("--test_center", type=str, default=None, 
                        help="If not None, uses leave-one-center-out cross-validation with the specified center as test. If None, uses k-fold cross-validation.")
    group.add_argument("--val_seed", type=int, default=0, 
                       help="The seed to use for validation split.")            

    group = parser.add_argument_group("TRAIN SET FILTERING", description="""Arguments for filtering the training set.""")
    group.add_argument("--undersample_benign_ratio", type=lambda x: float(x) if not x.lower() == 'none' else None, default=None,
                       help="""If not None, undersamples benign cores with the specified ratio. E.g if `--undersample_benign_ratio=3`, 
                       we will have 3 benign cores for every malignant core.""")
    group.add_argument("--min_involvement_train", type=float, default=0.0,
                       help="""The minimum involvement (percentage of cancer by tissue area) to use for training. If not zero, should be a percentage (e.g.`--min_involvement_train=40`)
                    we found discarding cores with <40 percent cancer from the training set improved the model by letting it focus on the stronger cancer signals.""")
    group.add_argument("--remove_benign_cores_from_positive_patients", action="store_true",
        help="""If True, removes benign cores from positive patients. This can help because these cores have a high chance of also containing malignancy outside the needle region""")
    group.add_argument("--limit_train_data", type=float, default=None, 
                       help="""If less than 1, chooses a center-balanced subset of the original train data to train with. The value given is the fraction of the original data to use.""")
    group.add_argument("--train_subsample_seed", type=int, default=42, help="The seed to use for subsampling the training data (if limit_train_data < 1).") 

    # fmt: on
    return parser.parse_args()


def main(args):
    metadata_table = pd.read_csv(args.data_csv_path)
    cohort_selector = CohortSelector(metadata_table)

    if args.fold is not None:
        train_patients, val_patients, test_patients = (
            cohort_selector.get_patient_splits_by_fold(
                fold=args.fold, n_folds=args.n_folds
            )
        )
    else:
        assert (
            args.test_center is not None
        ), "Either fold or test_center must be specified"
        train_patients, val_patients, test_patients = (
            cohort_selector.get_patient_splits_by_center(leave_out=args.test_center)
        )

    train_cores = cohort_selector.get_core_ids(train_patients)
    val_cores = cohort_selector.get_core_ids(val_patients)
    test_cores = cohort_selector.get_core_ids(test_patients)

    train_cores = cohort_selector.apply_core_filters(
        train_cores,
        exclude_benign_cores_from_positive_patients=args.remove_benign_cores_from_positive_patients,
        involvement_threshold_pct=args.min_involvement_train,
        undersample_benign_ratio=args.undersample_benign_ratio,
    )

    if args.limit_train_data is not None:
        assert args.limit_train_data < 1, "limit_train_data should be less than 1"

        cores = train_cores
        center = [core.split("-")[0] for core in cores]

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - args.limit_train_data,
            random_state=args.train_subsample_seed,
        )
        for train_index, _ in sss.split(cores, center):
            train_cores = [cores[i] for i in train_index]

    output = {
        "train": train_cores,
        "val": val_cores,
        "test": test_cores,
    }
    output["fold"] = args.fold
    output["n_folds"] = args.n_folds
    output["test_center"] = args.test_center
    output["val_seed"] = args.val_seed
    output["undersample_benign_ratio"] = args.undersample_benign_ratio
    output["min_involvement_train"] = args.min_involvement_train
    output["remove_benign_cores_from_positive_patients"] = (
        args.remove_benign_cores_from_positive_patients
    )
    output["limit_train_data"] = args.limit_train_data
    output["train_subsample_seed"] = args.train_subsample_seed

    with open(args.output_filename, "w") as f:
        print(f"Saving splits to {args.output_filename}")
        print(f"Train: {len(train_cores)} cores")
        print(f"Val: {len(val_cores)} cores")
        print(f"Test: {len(test_cores)} cores")
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main(parse_args())

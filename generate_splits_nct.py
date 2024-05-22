from dataclasses import dataclass
import os
from numpy import array_split
from simple_parsing import ArgumentParser, parse, Serializable
import json
from rich_argparse import ArgumentDefaultsRichHelpFormatter
import pandas as pd
from src.dataset import CohortSelector
from sklearn.model_selection import StratifiedShuffleSplit
from config import CohortSelectionOptions, DataPaths


@dataclass
class Args(Serializable):
    """Arguments for cohort selection"""

    data_paths: DataPaths = DataPaths()
    cohort_selection: CohortSelectionOptions = CohortSelectionOptions()
    output_filename: str = (
        "splits.json"  # name of a json file where the splits will be dumped
    )


def main(args: Args):

    metadata_table = pd.read_csv(args.data_paths.metadata_csv_path)
    cohort_selector = CohortSelector(metadata_table)
    train_cores, val_cores, test_cores, ssl_train_cores = (
        cohort_selector.select_cohort_from_options(args.cohort_selection)
    )

    output = {
        "train": train_cores,
        "val": val_cores,
        "test": test_cores,
        "ssl_train": ssl_train_cores,
    }

    os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)

    with open(args.output_filename, "w") as f:
        print(f"Saving splits to {args.output_filename}")
        print(f"Train: {len(train_cores)} cores")
        print(f"Val: {len(val_cores)} cores")
        print(f"Test: {len(test_cores)} cores")
        json.dump(output, f, indent=4)

    with open(args.output_filename.replace(".json", "_params.json"), "w") as f:
        args.cohort_selection.dump_json(f)


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(Args, dest="args")
    main(parser.parse_args().args)

    # main(parse(Args, add_config_path_arg=True))

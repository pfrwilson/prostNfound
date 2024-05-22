from dataclasses import dataclass
from simple_parsing import parse
import pandas as pd
import numpy as np
from config import DataPaths


@dataclass
class Args: 
    """Args for generating prompt table.
    
    Args:
        output_filename (str): Output filename (.csv) where the splits will be saved
        paths (DataPaths): Data paths
    """
    output_filename: str = "prompt_table.csv"
    paths: DataPaths = DataPaths()


# encode the core location to an index, 
# left and right get mapped to same code
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


def main(args: Args):
    """Make prompt table"""

    # load metadata
    table = pd.read_csv(args.paths.metadata_csv_path, index_col=0)

    # make prompt_table
    prompt_table = pd.DataFrame()
    
    # get core ids
    prompt_table["core_id"] = table["core_id"]
    
    # normalize the floating point prompts
    prompt_table["normalized_psa"] = (
        (table["psa"] - table["psa"].min()) / (table["psa"].max() - table["psa"].min())
    ).fillna(table["psa"].mean())
    
    prompt_table["normalized_psadensity"] = (
        (table["approx_psa_density"] - table["approx_psa_density"].min())
        / (table["approx_psa_density"].max() - table["approx_psa_density"].min())
    ).fillna(table["approx_psa_density"].mean())
    
    prompt_table["normalized_age"] = (
        (table["age"] - table["age"].min()) / (table["age"].max() - table["age"].min())
    ).fillna(table["age"].mean())
    
    # encode the discrete prompts
    prompt_table["encoded_family_history"] = table["family_history"].apply(
        lambda t: 1 if t == True else (0 if t == False else 2)
    )
    
    prompt_table["encoded_core_location"] = table["loc"].apply(
        lambda t: CORE_LOCATION_TO_IDX[t]
    )

    # check for NaN values
    for column in prompt_table.columns:
        if column != "core_id":
            assert not np.any(np.isnan(prompt_table[column].values))

    # save the prompt table
    prompt_table.to_csv(args.output_filename, index=False)


if __name__ == "__main__":
    args = parse(Args)
    main(args)

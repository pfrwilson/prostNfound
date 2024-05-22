from enum import StrEnum
from textwrap import indent
from turtle import back
from typing import Literal
from simple_parsing import parse, field
from simple_parsing.helpers.serialization.serializable import SerializableMixin, Serializable
from dataclasses import dataclass
import dotenv
import rich 
import os
from src.dataset import CohortSelectionOptions
from src.prostnfound import ProstNFoundConfig

dotenv.load_dotenv()


@dataclass(frozen=True)
class DataPaths: 
    """Data paths"""
    metadata_csv_path: str | None = os.getenv('PROSTNFOUND_DATA_METADATA_PATH')
    data_dir: str | None = os.getenv("PROSTNFOUND_DATA_DIR")
    data_h5_path: str | None = os.getenv("PROSTNFOUND_DATA_H5_PATH")



@dataclass(frozen=True)
class MainDataOptions: 
    """Config for prostNfound data"""
    prompt_table_csv_path: str | None = None # path to table containing floating point and discrete prompts
    num_workers: int = 4
    batch_size: int = 4 # The batch size to use for training. Often limited by GPU size - if you want a larger effective batch size you can also adjust `--accumulate_grad_steps`.
    augmentations: str | None = field(default=None, choices=['translate']) # The augmentations to use for training. We found random translation to boost performance compared to no augmentations.
    image_size: int = 1024 # The size to use for the images.
    mask_size: int = 256
    rf_as_bmode: bool = False # If True, uses the RF images as B-mode images. (Hack to be used to test foundation model performance on RF directly)


@dataclass
class WandbOptions: 
    """Config for wandb"""
    project: str = "miccai2024"
    group: str | None = None
    name: str | None = None
    log_images: bool = False
    tags: list[str] = field(default_factory=lambda:[])


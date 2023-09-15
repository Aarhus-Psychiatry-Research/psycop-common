"""Load and compare train and validation splits."""
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import wandb
import hydra

from psycop.common.model_training.data_loader.utils import (
    load_and_filter_split_from_cfg,
)
from psycop.common.model_training.config_schemas.conf_utils import (
    convert_omegaconf_to_pydantic_object,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "model_training" / "config"
REPO_ROOT = Path(__file__).resolve().parents[4]
SYNTH_DATA_PATH = REPO_ROOT / "common" / "test_utils" / "test_data" / "flattened" / "synth_flattened_with_outcome.csv"


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="default_config",
    version_base="1.2",
)
def main(cfg: FullConfigSchema, synth_data: bool = True):
    """Main."""
    cfg = convert_omegaconf_to_pydantic_object(cfg)

    if sys.platform == "win32":
        (PROJECT_ROOT / "wandb" / "debug-cli.onerm").mkdir(exist_ok=True, parents=True)

    if synth_data:

        df = pd.read_csv(SYNTH_DATA_PATH)

    train_df = load_and_filter_split_from_cfg(
        data_cfg=cfg.data,
        pre_split_cfg=cfg.preprocessing.pre_split,
        split="train",
    )

    val_df = load_and_filter_split_from_cfg(
        data_cfg=cfg.data,
        pre_split_cfg=cfg.preprocessing.pre_split,
        split="train",
    )

    return train_df, val_df


if __name__ == "__main__":
    main()
    print('Done!')

from collections.abc import Sequence
from typing import Optional

import pandas as pd
from timeseriesflattener.feature_specs.single_specs import AnySpec

from psycop.common.feature_generation.application_modules.flatten_dataset import (
    flatten_dataset_to_disk,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.model_training.application_modules.train_model.main import train_model
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema


def minimal_pipeline(
    project_info: ProjectInfo,
    feature_specs: list[AnySpec],
    prediction_times_df: pd.DataFrame,
    model_training_cfg: FullConfigSchema,
    add_birthdays: bool = True,
    split2ids_df: Optional[dict[str, pd.DataFrame]] = None,
    split_names: Sequence[str] = ("train", "val", "test"),
) -> float:
    flatten_dataset_to_disk(
        project_info=project_info,
        feature_specs=feature_specs,
        add_birthdays=add_birthdays,
        prediction_times_df=prediction_times_df,
        feature_set_dir=project_info.flattened_dataset_dir,
        split2ids_df=split2ids_df,
        split_names=split_names,
    )

    auroc = train_model(cfg=model_training_cfg)

    return auroc

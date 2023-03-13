from pathlib import Path

from psycop_model_training.config_schemas.data import DataSchema
from psycop_model_training.config_schemas.preprocessing import (
    PreSplitPreprocessingConfigSchema,
)
from psycop_model_training.data_loader.data_loader import DataLoader
from psycop_model_training.preprocessing.pre_split.full_processor import (
    process_full_dataset,
)

if __name__ == "__main__":
    dataset_path = Path(
        "E:/shared_resources/t2d/feature_sets/psycop_t2d_adminmanber_features_2023_03_09_18_37",
    )

    data_cfg = DataSchema(dir=dataset_path)
    dataset = DataLoader(data_cfg=data_cfg).load_dataset_from_dir(split_names=["train"])

    processed_dataset = process_full_dataset(
        dataset=dataset,
        pre_split_cfg=PreSplitPreprocessingConfigSchema(),
    )

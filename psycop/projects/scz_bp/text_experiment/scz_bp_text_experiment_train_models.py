from collections.abc import Iterable
from pathlib import Path

from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry


def regex_match_string_anywhere(input_str: str) -> str:
    return f".*{input_str}.*"


def get_list_of_paths_to_splits(feature_set_dir: Path) -> Iterable[str]:
    return [
        str(feature_set_dir / "train.parquet"),
        str(feature_set_dir / "val.parquet"),
        str(feature_set_dir / "test.parquet"),
    ]


if __name__ == "__main__":
    populate_baseline_registry()
    cfg_file = Path(__file__).parent / "text_exp_config.cfg"

    cfg = Config().from_disk(cfg_file)
    file_dir = cfg["trainer"]["training_data"]["paths"][0]
    for feature_set_dir in Path(file_dir).iterdir():
        if not feature_set_dir.is_dir():
            print(f"{feature_set_dir} is not a directory. Skipping")
            continue
        # if "aktuelt_psykisk" in feature_set_dir.name:
        #     print(f"Skipping {feature_set_dir}")
        #     continue
        # if not "tfidf-1000" in feature_set_dir.name:
        #     print(f"Skipping {feature_set_dir}")
        #     continue
        cfg_copy = cfg.copy()
        cfg_copy["trainer"]["training_data"]["paths"] = get_list_of_paths_to_splits(
            feature_set_dir=feature_set_dir
        )

        cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg_copy))
        cfg_schema.logger.info(f"Training {feature_set_dir.name}")
        result = cfg_schema.trainer.train()

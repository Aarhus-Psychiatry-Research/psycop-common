from collections.abc import Iterable
from pathlib import Path

from confection import Config
from joblib import Parallel, delayed

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


def train_text_model(cfg: Config) -> float:
    populate_baseline_registry()

    cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg))
    cfg_schema.logger.info(f'Training {cfg["trainer"]["training_data"]["paths"]}')
    result = cfg_schema.trainer.train()
    return result.metric.value


if __name__ == "__main__":
    cfg_file = Path(__file__).parent / "text_exp_config.cfg"

    cfg = Config().from_disk(cfg_file)
    file_dir = cfg["trainer"]["training_data"]["paths"][0]

    cfgs = []
    for feature_set_dir in Path(file_dir).iterdir():
        if not feature_set_dir.is_dir():
            print(f"{feature_set_dir} is not a directory. Skipping")
            continue
        cfg_copy = cfg.copy()

        cfg_copy["trainer"]["training_data"]["paths"] = get_list_of_paths_to_splits(
            feature_set_dir=feature_set_dir
        )
        cfgs.append(cfg_copy)

    # add keyword config (made with tsflattener v2)
    keyword_cfg = Config().from_disk(Path(__file__).parent / "text_exp_keywords.cfg")
    cfgs.append(keyword_cfg)

    Parallel(n_jobs=9)(delayed(train_text_model)(cfg=filled_cfg) for filled_cfg in cfgs)

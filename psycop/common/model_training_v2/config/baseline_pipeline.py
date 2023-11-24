from pathlib import Path

from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema


def train_baseline_model(cfg_file: Path) -> float:
    cfg_raw = Config().from_disk(cfg_file).to_str()
    resolved = BaselineRegistry.resolve(cfg_raw)
    cfg_schema = BaselineSchema(**resolved)

    cfg_schema.logger.log_config(
        cfg_raw.to_str(),
    )  # Dict handling, might have to be flattened depending on the logger. Probably want all loggers to take flattened dicts.


def train_baseline_model_from_schema(cfg: BaselineSchema) -> float:
    cfg.logger.log_config(
        cfg.dict(),
    )  # Dict handling, might have to be flattened depending on the logger. Probably want all loggers to take flattened dicts.
    # TODO: Currently logs the resolved objects. We want to fix that.

    result = cfg.trainer.train()
    result.df.write_parquet(cfg.project_info.experiment_path / "eval_df.parquet")
    # TODO: https://github.com/Aarhus-Psychiatry-Research/psycop-common/issues/447 Allow dynamic generation of experiments paths

    return result.metric.value

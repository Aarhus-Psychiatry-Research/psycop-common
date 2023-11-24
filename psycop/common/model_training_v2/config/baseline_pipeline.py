from pathlib import Path

from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema


def train_baseline_model(cfg_file: Path) -> float:
    cfg_raw = BaselineRegistry.fill(Config().from_disk(cfg_file))
    cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg_raw))

    cfg_schema.logger.log_config(
        cfg_raw,
    )

    return train_baseline_model_from_schema(cfg_schema)


def train_baseline_model_from_schema(cfg: BaselineSchema) -> float:
    result = cfg.trainer.train()
    result.df.write_parquet(cfg.project_info.experiment_path / "eval_df.parquet")
    # TODO: https://github.com/Aarhus-Psychiatry-Research/psycop-common/issues/447 Allow dynamic generation of experiments paths

    return result.metric.value

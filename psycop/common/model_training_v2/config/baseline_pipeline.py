from pathlib import Path

from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema

from .config_utils import load_baseline_config


def train_baseline_model_from_file(cfg_file: Path) -> float:
    schema = load_baseline_config(cfg_file)
    return train_baseline_model_from_schema(schema)


def train_baseline_model_from_schema(cfg: BaselineSchema) -> float:
    cfg.logger.log_config(cfg.model_dump())
    result = cfg.trainer.train()
    result.df.write_parquet(cfg.project_info.experiment_path / "eval_df.parquet")
    return result.metric.value

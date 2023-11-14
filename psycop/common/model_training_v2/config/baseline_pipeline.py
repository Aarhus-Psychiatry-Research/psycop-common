from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema


def train_baseline_model(cfg: BaselineSchema) -> float:
    cfg.logger.log_config(
        cfg.dict(),
    )  # Dict handling, might have to be flattened depending on the logger. Probably want all loggers to take flattened dicts.

    result = cfg.trainer.train()
    result.df.write_parquet(cfg.project_info.experiment_path / "eval_df.parquet")

    return result.metric.value

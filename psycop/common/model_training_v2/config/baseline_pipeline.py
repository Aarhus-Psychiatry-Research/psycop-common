from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema


def train_baseline_model(cfg: BaselineSchema) -> float:
    cfg.logger.log_config(
        cfg.dict(),
    )  # Dict handling, might have to be flattened depending on the logger. Probably want all loggers to take flattened dicts.
    # TODO: Currently logs the resolved objects. We want to fix that.

    result = cfg.trainer.train()
    result.df.write_parquet(cfg.project_info.experiment_path / "eval_df.parquet")
    # TODO: https://github.com/Aarhus-Psychiatry-Research/psycop-common/issues/447 Allow dynamic generation of experiments paths

    return result.metric.value

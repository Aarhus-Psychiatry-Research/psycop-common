from psycop.common.sequence_models.loggers.wandb_logger import WandbLogger


def test_wandb_logger():
    logger = WandbLogger(
        project_name="test",
        group="test_group",
        entity="test_entity",
        run_name="test_run",
    )
    logger.log_hyperparams({"test_param": "test_value"})

    # Should be able to log metrics multiple times
    logger.log_metrics({"test_metric": 0.5})
    logger.log_metrics({"test_metric": 0.5})

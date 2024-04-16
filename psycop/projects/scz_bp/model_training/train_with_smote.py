from pathlib import Path

from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric
from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import populate_scz_bp_registry

if __name__ == "__main__":
    populate_baseline_registry()
    populate_scz_bp_registry()  # noqa: ERA001

    cfg_file = (
        Path(__file__).parent
        / "config"
        / "augmentation"
        / "scz_bp_structured_text_best_xgboost_smote.cfg"
    )
    cfg = Config().from_disk(cfg_file)

    base_positive_samples = 14914

    for multiply_samples in range(5, 11):
        print(f"Sample multiplier: {multiply_samples}")
        cfg_copy = cfg.copy()
        cfg_copy["trainer"]["task"]["task_pipe"]["sklearn_pipe"]["*"]["smote"][
            "n_minority_samples"
        ] = base_positive_samples * multiply_samples

        cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg_copy))
        cfg_schema.logger.log_config(cfg_copy)
        cfg_schema.logger.log_metric(
            CalculatedMetric(name="sample_multiplier", value=multiply_samples)
        )

        result = cfg_schema.trainer.train()

from pathlib import Path

from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.loggers.dummy_logger import DummyLogger
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)

if __name__ == "__main__":
    pred_times = SczBpCohort.get_filtered_prediction_times_bundle()

    cfg = Config().from_disk(Path(__file__).parent / "flowchart_config.cfg")

    populate_baseline_registry()
    data = BaselineRegistry.resolve({"data": cfg["trainer"]["training_data"]})["data"].load()

    preprocessing_pipeline = BaselineRegistry().resolve(
        {"pipe": cfg["trainer"]["preprocessing_pipeline"]}
    )["pipe"]
    preprocessing_pipeline._logger = DummyLogger()
    # step through with debugger
    preprocessing_pipeline.apply(data)

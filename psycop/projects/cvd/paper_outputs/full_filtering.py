# %%
# Get cohort data
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.config_utils import resolve_and_fill_config
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import (
    BaselinePreprocessingPipeline,
)
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    cvd_pred_filtering,
    cvd_pred_times,
)

prediction_times = cvd_pred_filtering()

# Get preprocessing pipeline from the run
run = MlflowClientWrapper().get_run("CVD", "CVD layer 1, base")

preprocessing_cfg = run.get_config()["trainer"]["preprocessing_pipeline"]
preprocessing_cfg["eager"] = True

# Delete all steps which are not filters
preprocessing_cfg["*"] = {k: v for k, v in preprocessing_cfg["*"].items() if "filter" in k}

pipeline: BaselinePreprocessingPipeline = BaselineRegistry.resolve(preprocessing_cfg)[
    "preprocessing"
]

# Apply preprocessing pipeline
preprocessed_data = pipeline.apply(prediction_times.prediction_times.frame.lazy())

# Output results of the preprocessing pipeline, step by step

# %%
# Get cohort data
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import (
    BaselinePreprocessingPipeline,
)
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    cvd_pred_filtering,
    cvd_pred_times,
)

prediction_times = cvd_pred_filtering()

# %%
# Get preprocessing pipeline from the run
run = MlflowClientWrapper().get_run("CVD", "CVD layer 1, base")

# %%
cfg = run.get_config()
cfg["trainer"]["preprocessing_pipeline"]["eager"] = True

# %%
train_baseline_model_from_cfg(cfg=cfg)

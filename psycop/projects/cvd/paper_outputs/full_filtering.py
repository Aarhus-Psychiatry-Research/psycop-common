# %%
%load_ext autoreload
%autoreload 2

# %%
# Get cohort data
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg

# %%
# Replicate preprocessing pipeline from run
run = MlflowClientWrapper().get_run("CVD", "CVD layer 1, base")

# %%
from pathlib import Path

cfg = run.get_config()



# %%

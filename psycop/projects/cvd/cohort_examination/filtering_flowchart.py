# %%
run = MlflowClientWrapper().get_run(experiment_name="CVD", run_name="CVD layer 1, base")
cfg = run.get_config()

# %%
##########################
# Preprocessing pipeline #
##########################
# %%
import pathlib
from tempfile import mkdtemp

tmp_cfg = pathlib.Path(mkdtemp()) / "tmp.cfg"
cfg.to_disk(tmp_cfg)

# %%
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.cvd.model_training.populate_cvd_registry import populate_with_cvd_registry

populate_baseline_registry()
populate_with_cvd_registry()

# %%
from psycop.common.model_training_v2.config.config_utils import resolve_and_fill_config

filled = resolve_and_fill_config(tmp_cfg, fill_cfg_with_defaults=True)

from psycop.common.model_training_v2.loggers.terminal_logger import TerminalLogger

pipeline: BaselinePreprocessingPipeline = filled["trainer"].preprocessing_pipeline
pipeline._logger = TerminalLogger() # type: ignore
pipeline.steps[0].split_to_keep = ["train", "val", "test"] # Do not filter by region

# %%
flattened_data = pl.scan_parquet(cfg["trainer"]["training_data"]["paths"][0]).lazy()

prior = len(flattened_data.collect())
print(f"Rows in CohortDefiner: {prior}")
for step in pipeline.steps:
    flattened_data = step.apply(flattened_data)
    post_step_rows = len(flattened_data.collect())

    diff = prior - post_step_rows
    print(f"Rows dropped by {step.__class__.__name__}: {diff}")
    prior = post_step_rows

outcome_col_name = cfg["trainer"]["outcome_col_name"]
with_cvd = flattened_data.filter(pl.col(outcome_col_name) == pl.lit(1)).collect()
without_cvd = flattened_data.filter(pl.col(outcome_col_name) == pl.lit(0)).collect()

print(f"End_rows: {len(flattened_data.collect())}")
print(f"With CVD: {len(with_cvd)}")
print(f"Without CVD: {len(without_cvd)}")
# %%

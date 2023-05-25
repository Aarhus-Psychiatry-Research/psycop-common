"""Main feature generation."""
# %%
import logging

from psycop.common.feature_generation.application_modules.project_setup import (
    get_project_info,
)

log = logging.getLogger()

# %%
project_info = get_project_info(
    project_name="t2d",
)

from psycop.projects.t2d.feature_generation.specify_features import FeatureSpecifier

feature_specs = FeatureSpecifier(
    project_info=project_info,
    min_set_for_debug=False,  # Remember to set to False when generating full dataset
).get_feature_specs()

selected_specs = [
    spec
    for spec in feature_specs
    if "pred" in spec.get_col_str() or "outc" in spec.get_col_str()
]

# %%
# %reload_ext autoreload
# %autoreload 2

# %%
from psycop.common.feature_generation.data_checks.flattened.feature_describer import (
    save_feature_descriptive_stats_from_dir,
)
from psycop.projects.t2d.paper_outputs.config import BEST_EVAL_PIPELINE, run.paper_outputs.paths.tables

out_dir = run.paper_outputs.paths.tables / "feature_description"
out_dir.mkdir(parents=True, exist_ok=True)

save_feature_descriptive_stats_from_dir(
    feature_set_dir=BEST_EVAL_PIPELINE.inputs._get_flattened_split_path(
        split="train"
    ).parent,
    feature_specs=selected_specs,  # type: ignore
    file_suffix="parquet",
    splits=["train"],
    out_dir=out_dir,
)

# %%

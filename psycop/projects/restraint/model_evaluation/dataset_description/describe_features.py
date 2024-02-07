"""Describe features"""

import logging

from psycop.projects.restraint.restraint_global_config import RESTRAINT_PROJECT_INFO

log = logging.getLogger()

from psycop.projects.restraint.feature_generation.modules.specify_features import FeatureSpecifier

feature_specs = FeatureSpecifier(
    project_info=RESTRAINT_PROJECT_INFO, min_set_for_debug=False
).get_feature_specs()

selected_specs = [
    spec
    for spec in feature_specs
    if "pred" in spec.get_output_col_name() or "outc" in spec.get_output_col_name()
]

from psycop.common.feature_generation.data_checks.flattened.feature_describer import (
    save_feature_descriptive_stats_from_dir,
)
from psycop.projects.restraint.model_evaluation.config import TEXT_EVAL_RUN, TEXT_TABLES_PATH

out_dir = TEXT_TABLES_PATH / "feature_description"
out_dir.mkdir(parents=True, exist_ok=True)

save_feature_descriptive_stats_from_dir(
    feature_set_dir=TEXT_EVAL_RUN._get_flattened_split_path(split="train").parent,  # type: ignore
    feature_specs=selected_specs,  # type: ignore
    file_suffix="parquet",
    splits=["train"],
    out_dir=out_dir,
)

"""Describe features"""

import logging

from psycop.common.feature_generation.application_modules.project_setup import (
    get_project_info,
)

log = logging.getLogger()

project_info = get_project_info(
    project_name="coercion",
)

from care_ml.feature_generation.modules.specify_features import FeatureSpecifier

feature_specs = FeatureSpecifier(
    project_info=project_info,
    min_set_for_debug=False,
).get_feature_specs()

selected_specs = [
    spec
    for spec in feature_specs
    if "pred" in spec.get_col_str() or "outc" in spec.get_col_str()
]

from care_ml.model_evaluation.config import TEXT_EVAL_RUN, TEXT_TABLES_PATH
from psycop.common.feature_generation.data_checks.flattened.feature_describer import (
    save_feature_descriptive_stats_from_dir,
)

out_dir = TEXT_TABLES_PATH / "feature_description"
out_dir.mkdir(parents=True, exist_ok=True)

save_feature_descriptive_stats_from_dir(
    feature_set_dir=TEXT_EVAL_RUN._get_flattened_split_path(split="train").parent,
    feature_specs=selected_specs,  # type: ignore
    file_suffix="parquet",
    splits=["train"],
    out_dir=out_dir,
)

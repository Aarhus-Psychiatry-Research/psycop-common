from pathlib import Path
from typing import AnyStr
from application.t2d.modules.project_setup import ProjectInfo

from psycop_feature_generation.data_checks.flattened.data_integrity import (
    save_feature_set_integrity_from_dir,
)
from psycop_feature_generation.data_checks.flattened.feature_describer import (
    save_feature_description_from_dir,
)


def save_feature_set_description_to_disk(
    feature_specs: list[_AnySpec],
    describe_splits: bool = True,
    compare_splits: bool = True,
    project_info: ProjectInfo,
):
    """Describe output.

    Args:
        feature_specs (list): List of feature specs
        out_dir (Path): Path to output dir.
        file_suffix (str): File suffix.
        describe_splits (bool, optional): Whether to describe each split. Defaults to True.
        compare_splits (bool, optional): Whether to compare splits, e.g. do all categories exist in both train and val. Defaults to True.
    """
    # Describe each spit individually. 
    for method in ["describe", "compare"]:
        if method == "describe" and describe_splits:
            splits = ["train"]
            specs_to_describe = [spec for spec in feature_specs if spec.prefix == project_info.prefix.predictor])],
            compare_splits_in_method = False
            out_dir = project_info.feature_set_path / "feature_set_description",
        
        if method == "compare" and compare_splits:
            splits = ["train", "val", "test"]
            specs_to_describe = None # Don't describe specs in comparison, to avoid leakage by describing features in val and test
            compare_splits_in_method = True
            out_dir = project_info.feature_set_path / "data_integrity",
            
        save_feature_set_integrity_from_dir(
                splits=splits,
                out_dir=out_dir,
                feature_specs=specs_to_describe,
                dataset_format=project_info.dataset_format,
                describe_splits=describe_splits,
                compare_splits=compare_splits_in_method,
            )
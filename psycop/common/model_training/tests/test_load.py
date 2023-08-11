"""Testing of loader functions."""


import pandas as pd
import pytest

from psycop.common.model_training.config_schemas.conf_utils import (
    validate_classification_objective,
)
from psycop.common.model_training.config_schemas.data import ColumnNamesSchema
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.data_loader.col_name_checker import (
    check_columns_exist_in_dataset,
)
from psycop.common.model_training.data_loader.utils import (
    load_and_filter_train_from_cfg,
)


def test_load_lookbehind_exceeds_lookbehind_threshold(
    muteable_test_config: FullConfigSchema,
):
    """Test that columns are dropped if their lookbehind are larger than the
    lookbehind threshold."""
    cfg = muteable_test_config

    n_cols_before_filtering = load_and_filter_train_from_cfg(cfg=cfg).shape[1]

    cfg.preprocessing.pre_split.lookbehind_combination = [30, 60]

    n_cols_after_filtering = load_and_filter_train_from_cfg(cfg=cfg).shape[1]

    assert n_cols_before_filtering - n_cols_after_filtering == 2


def test_load_lookbehind_not_in_lookbehind_combination(
    muteable_test_config: FullConfigSchema,
):
    """Test that columns are dropped if their lookbehind is not in the
    specified lookbehind combination list."""
    cfg = muteable_test_config

    n_cols_before_filtering = load_and_filter_train_from_cfg(cfg=cfg).shape[1]

    cfg.preprocessing.pre_split.lookbehind_combination = [60]
    n_cols_after_filtering = load_and_filter_train_from_cfg(cfg=cfg).shape[1]

    assert n_cols_before_filtering - n_cols_after_filtering == 3


def test_check_columns_exist_in_dataset():
    """Test that the check_columns_exist_in_dataset function raises an error if
    a column is missing from the dataframe."""

    test_schema = ColumnNamesSchema(
        pred_timestamp="pred_timestamp",
        outcome_timestamp="outcome_timestamp",
        id="id",
        age="age",
        is_female="is_female",
        exclusion_timestamp="exclusion_timestamp",
        custom_columns=["custom1", "custom2"],
    )

    df = pd.DataFrame(
        {
            "pred_timestamp": [1, 2, 3],
            "pred_time_uuid": [1, 2, 3],
            "outcome_timestmamp": [4, 5, 6],
            "id": [7, 8, 9],
            "age": [10, 11, 12],
            "is_female": [13, 14, 15],
            "exclusion_timestamp": [16, 17, 18],
            "custom1": [19, 20, 21],
        },
    )

    with pytest.raises(ValueError, match="custom2"):
        check_columns_exist_in_dataset(col_name_schema=test_schema, df=df)


def test_validate_classification_objective(muteable_test_config: FullConfigSchema):
    cfg = muteable_test_config

    if cfg.preprocessing.pre_split.classification_objective == "binary":
        with pytest.raises(
            ValueError,
            match="Only one outcome column can be used for binary classification tasks.",
        ):
            validate_classification_objective(
                cfg=cfg,
                col_names=["outc_event1", "outc_event2"],
            )

        assert (
            validate_classification_objective(cfg=cfg, col_names="outc_event1") is None
        )

    elif cfg.preprocessing.pre_split.classification_objective == "multilabel":
        with pytest.raises(
            ValueError,
            match="Multiple outcome columns are needed for multilabel classification tasks.",
        ):
            validate_classification_objective(cfg=cfg, col_names="outc_event1")

        assert (
            validate_classification_objective(
                cfg=cfg,
                col_names=["outc_event1", "outc_event2"],
            )
            is None
        )

# pyright: reportPrivateUsage=false
from pathlib import Path

import pandas as pd
import pytest

from psycop.common.model_training.config_schemas.data import DataSchema
from psycop.common.model_training.data_loader.data_loader import DataLoader
from psycop.common.test_utils.str_to_df import str_to_df

## write test for hashing of uuids


@pytest.fixture
def dataloader() -> DataLoader:
    data_cfg = DataSchema(dir=Path(), suffix="", splits_for_training=[""], n_training_samples=None)
    return DataLoader(data_cfg=data_cfg)


@pytest.mark.parametrize(
    ("feature_df", "expected"),
    [("feature_df_same_order_uuids", True), ("feature_df_different_order_uuids", False)],
)
def test_check_dataframes_can_be_concatenated(
    base_feature_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    expected: bool,
    request: pytest.FixtureRequest,  # pytest fixture
):
    feature_df = request.getfixturevalue(feature_df)  # type: ignore
    assert (
        DataLoader._check_dataframes_can_be_concatenated(
            datasets=[base_feature_df, feature_df], uuid_column="prediction_time_uuid"
        )
        is expected
    )


def test_check_and_merge_feature_sets_concatenated_correct_output(
    base_feature_df: pd.DataFrame, feature_df_same_order_uuids: pd.DataFrame, dataloader: DataLoader
):
    concatenated_df = dataloader._check_and_merge_feature_sets(  #
        datasets=[base_feature_df, feature_df_same_order_uuids]
    )

    expected_cols = {
        "prediction_time_uuid",
        "feature_name_1",
        "feature_name_2",
        "dw_ek_borger",
        "timestamp",
    }

    assert set(concatenated_df.columns) == expected_cols
    assert len(concatenated_df.columns) == len(expected_cols)


def test_check_and_merge_feature_sets(
    base_feature_df: pd.DataFrame,
    feature_df_same_order_uuids: pd.DataFrame,
    feature_df_different_order_uuids: pd.DataFrame,
    dataloader: DataLoader,
):
    concatenated_df = dataloader._check_and_merge_feature_sets(
        datasets=[base_feature_df, feature_df_same_order_uuids]
    )
    joined_df = dataloader._check_and_merge_feature_sets(
        datasets=[base_feature_df, feature_df_different_order_uuids]
    )

    assert concatenated_df.equals(joined_df)


def test_check_and_merge_feature_sets_too_many_rows(
    base_feature_df: pd.DataFrame, dataloader: DataLoader
):
    feature_df_too_many_rows = str_to_df(
        """prediction_time_uuid,feature_name_2,dw_ek_borger,timestamp
x_2010,2,x,2010
x_2011,2,x,2011
y_2010,3,y,2010
y_2011,3,y,2011
z_2010,4,z,2010"""
    )
    with pytest.raises(ValueError, match="The datasets have a different amount of rows.*"):
        dataloader._check_and_merge_feature_sets(
            datasets=[base_feature_df, feature_df_too_many_rows]
        )


def test_check_and_merge_feature_sets_not_matching_uuids(
    base_feature_df: pd.DataFrame, dataloader: DataLoader
):
    feature_df_not_matching_uuids = str_to_df(
        """prediction_time_uuid,feature_name_2,dw_ek_borger,timestamp
x_2010,0,x,2010
x_2011,0,x,2011
z_2010,1,z,2010
z_2011,1,z,2011"""
    )
    with pytest.raises(
        ValueError,
        match="The datasets have different uuids. Ensure that they have been created with the same prediction times.",
    ):
        dataloader._check_and_merge_feature_sets(
            datasets=[base_feature_df, feature_df_not_matching_uuids]
        )

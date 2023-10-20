# pyright: reportPrivateUsage=false
import pandas as pd
import pytest

from psycop.common.model_training.data_loader.data_loader import DataLoader
from psycop.common.test_utils.str_to_df import str_to_df

## write test for hashing of uuids


## write test for merging feature sets
@pytest.fixture()
def base_feature_df() -> pd.DataFrame:
    return str_to_df(
        """uuid,feature_name_1
x_2010,0
x_2011,0
y_2010,1,
y_2011,1""",
    )


@pytest.fixture()
def feature_df_same_order_uuids() -> pd.DataFrame:
    return str_to_df(
        """uuid,feature_name_2
x_2010,2
x_2011,2
y_2010,3,
y_2011,3""",
    )


@pytest.fixture()
def feature_df_different_order_uuids() -> pd.DataFrame:
    return str_to_df(
        """uuid,feature_name_2
y_2010,3,
y_2011,3,
x_2010,2
x_2011,2""",
    )


def test_check_dataframes_can_be_concatenated(
    base_feature_df: pd.DataFrame,
    feature_df_same_order_uuids: pd.DataFrame,
):
    assert DataLoader._check_dataframes_can_be_concatenated(  #
        datasets=[base_feature_df, feature_df_same_order_uuids],
        uuid_column="uuid",
    )


def test_check_dataframes_can_be_concatenated_false(
    base_feature_df: pd.DataFrame,
    feature_df_different_order_uuids: pd.DataFrame,
):
    with pytest.raises(AssertionError):
        assert DataLoader._check_dataframes_can_be_concatenated(
            datasets=[base_feature_df, feature_df_different_order_uuids],
            uuid_column="uuid",
        )


def test_check_and_merge_feature_sets(
    base_feature_df: pd.DataFrame,
    feature_df_same_order_uuids: pd.DataFrame,
    feature_df_different_order_uuids: pd.DataFrame,
):
    concatenated_df = DataLoader._check_and_merge_feature_sets(
        datasets=[base_feature_df, feature_df_same_order_uuids],
        uuid_column="uuid",
    )
    joined_df = DataLoader._check_and_merge_feature_sets(
        datasets=[base_feature_df, feature_df_different_order_uuids],
        uuid_column="uuid",
    )

    assert concatenated_df.equals(joined_df)


def test_check_and_merge_feature_sets_too_many_rows(base_feature_df: pd.DataFrame):
    feature_df_too_many_rows = str_to_df(
        """uuid,feature_name_2
x_2010,2
x_2011,2
y_2010,3,
y_2011,3,
z_2010,4,""",
    )
    with pytest.raises(
        ValueError,
        match="The datasets have a different amount of rows. Ensure that they have been created with the same prediction times.",
    ):
        DataLoader._check_and_merge_feature_sets(
            datasets=[base_feature_df, feature_df_too_many_rows],
            uuid_column="uuid",
        )


def test_check_and_merge_feature_sets_not_matching_uuids(base_feature_df: pd.DataFrame):
    feature_df_not_matching_uuids = str_to_df(
        """uuid,feature_name_2
x_2010,0
x_2011,0
z_2010,1,
z_2011,1""",
    )
    with pytest.raises(
        ValueError,
        match="The datasets have different uuids. Ensure that they have been created with the same prediction times.",
    ):
        DataLoader._check_and_merge_feature_sets(
            datasets=[base_feature_df, feature_df_not_matching_uuids],
            uuid_column="uuid",
        )

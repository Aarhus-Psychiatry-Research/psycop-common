import polars as pl
import pytest
from polars.testing import assert_frame_equal

from psycop.common.model_training_v2.trainer.data.data_filters.geographical_split.make_geographical_split import (
    add_migration_date_by_patient,
    add_shak_to_region_mapping,
    get_first_visit_at_each_region_by_patient,
    get_first_visit_at_second_region_by_patient,
    get_first_visit_by_patient,
    load_shak_to_location_mapping,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


@pytest.fixture()
def mock_geography_data() -> pl.DataFrame:
    return str_to_pl_df(
        """dw_ek_borger,timestamp,region
        1, 2020-01-01,vest
        1, 2020-02-01,vest
        1, 2020-03-01,midt
        2, 2020-01-01,vest
        3, 2020-01-01,øst
        3, 2020-02-01,øst
        4, 2020-01-01,vest
        4, 2020-02-01,vest
        4, 2020-03-01,vest
        4, 2020-04-01,vest
""",
    )


def test_add_shak_to_region_mapping():
    visits_df = str_to_pl_df(
        """dw_ek_borger,timestamp,shak_location
        1, 2020-01-01, 660010 # øst
        2, 2020-01-01, 660061 # vest
        3, 2020-01-01, 660005 # midt
        4, 2020-01-01, 660011 # dropped""",
    ).with_columns(pl.col("shak_location").cast(str))

    shak_to_location_mapping_df = load_shak_to_location_mapping()
    shak_codes_to_drop = ["660011"]

    visits_with_region = add_shak_to_region_mapping(
        visits=visits_df,
        shak_to_location_df=shak_to_location_mapping_df,
        shak_codes_to_drop=shak_codes_to_drop,
    )

    expected_df = str_to_pl_df(
        """dw_ek_borger,timestamp,region
        1,2020-01-01,øst
        2,2020-01-01,vest
        3,2020-01-01,midt""",
    )
    assert_frame_equal(visits_with_region, expected_df)


def test_get_first_visit_at_each_region_by_patient(mock_geography_data: pl.DataFrame):
    expected = str_to_pl_df(
        """dw_ek_borger,timestamp,region
        1,2020-01-01,vest
        1,2020-03-01,midt
        2,2020-01-01,vest
        3,2020-01-01,øst
        4,2020-01-01,vest""",
    )

    first_visit_at_each_region_df = get_first_visit_at_each_region_by_patient(
        df=mock_geography_data,
    )
    # select to reorder colunns to match output
    assert_frame_equal(
        first_visit_at_each_region_df,
        expected.select("dw_ek_borger", "region", "timestamp"),
    )


def test_get_first_visit_by_patient(mock_geography_data: pl.DataFrame):
    expected = str_to_pl_df(
        """dw_ek_borger,timestamp,region
        1,2020-01-01,vest
        2,2020-01-01,vest
        3,2020-01-01,øst
        4,2020-01-01,vest""",
    )
    first_visit_at_each_region_df = get_first_visit_by_patient(df=mock_geography_data)

    assert_frame_equal(first_visit_at_each_region_df.sort("dw_ek_borger"), expected)


def test_get_first_visit_at_second_region_by_patient(mock_geography_data: pl.DataFrame):
    expected = str_to_pl_df(
        """dw_ek_borger,first_regional_move_timestamp,second_region
        1,2020-03-01,midt""",
    )

    first_visit_at_each_region_df = get_first_visit_at_each_region_by_patient(
        df=mock_geography_data,
    )
    first_visit_at_second_region_df = get_first_visit_at_second_region_by_patient(
        df=first_visit_at_each_region_df,
    )
    # select to reorder colunns to match output
    assert_frame_equal(
        first_visit_at_second_region_df,
        expected.select(
            "dw_ek_borger",
            "second_region",
            "first_regional_move_timestamp",
        ),
    )


def test_add_migration_date_for_each_patient():
    first_visit_at_first_region_df = str_to_pl_df(
        """dw_ek_borger,timestamp,region
        1,2020-01-01,vest
        2,2020-01-01,midt""",
    )

    first_visit_at_second_region_df = str_to_pl_df(
        """dw_ek_borger,first_regional_move_timestamp,second_region
        1,2020-03-01,midt""",
    )

    expected = str_to_pl_df(
        """dw_ek_borger,region,first_regional_move_timestamp,second_region
        1,vest,2020-03-01,midt
        2,midt,2100-01-01,null,""",
    ).with_columns(
        pl.col("first_regional_move_timestamp").cast(pl.Datetime(time_unit="us")),
    )

    migration_date_df = add_migration_date_by_patient(
        first_visit_at_first_region_df=first_visit_at_first_region_df,
        first_visit_at_second_region_df=first_visit_at_second_region_df,
    )

    assert_frame_equal(migration_date_df, expected)

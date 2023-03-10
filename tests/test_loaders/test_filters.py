from psycop_feature_generation.loaders.filters.diabetes_filters import (
    keep_rows_where_diag_matches_t2d_diag,
)
from psycop_ml_utils.utils_for_testing import str_to_df


def test_keep_rows_where_diag_matches_t2d_diag():
    test_df = str_to_df(
        """diagnosegruppestreng,timestamp,dw_ek_borger,keep,
            A:DE14#+:ALFC3, 2021-06-30, 1, 1,
            A:DE14#+:DE162, 2021-05-30, 1, 1,
            A, 2021-04-30, 1, 0,
            A:DG632, 2021-04-30, 1, 1""",
    )

    df = keep_rows_where_diag_matches_t2d_diag(
        df=test_df,
        col_name="diagnosegruppestreng",
    )

    assert df["keep"].mean() == 1.0

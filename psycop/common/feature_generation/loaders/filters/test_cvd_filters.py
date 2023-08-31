from psycop.common.feature_generation.loaders.filters.cvd_filters import (
    keep_rows_where_diag_matches_cvd_diag,
)
from psycop.common.test_utils.str_to_df import str_to_df


def test_cvd_diagnosegruppestreng_filtering():
    df = str_to_df(
        """
    diagnosegruppestreng,keep,
    DI21,Y, # Keep, exact match
    DI211,Y, # Keep, substring match
    DI62,N, # Drop, excluded
    DI67.1,N, # Drop, excluded
    DI67.5,N, # Drop, excluded
    DI68.2,N, # Drop, excluded
                   """,
    )

    filtered_df = keep_rows_where_diag_matches_cvd_diag(
        df=df,
        col_name="diagnosegruppestreng",
    )

    assert all(keep == "Y" for keep in filtered_df["keep"].tolist())
    assert len(filtered_df) == len(filtered_df[filtered_df["keep"] == "Y"])

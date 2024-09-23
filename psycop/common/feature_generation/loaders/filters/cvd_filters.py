import pandas as pd

from psycop.common.feature_generation.loaders.filters.diabetes_filters import (
    keep_rows_where_col_name_matches_pattern,
)


def only_SCORE2_CVD_diagnoses(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Keep rows where the diagnosis matches the T2D diagnosis.
    """
    cvd_regex_pattern = r"(DI2[1-3].*)|(DI6[0-9].*)"
    rows_matching_categories = keep_rows_where_col_name_matches_pattern(
        df=df, col_name=col_name, regex_pattern=cvd_regex_pattern
    )
    return rows_matching_categories

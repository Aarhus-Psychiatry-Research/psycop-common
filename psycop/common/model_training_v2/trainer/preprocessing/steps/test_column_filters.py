from psycop.common.model_training_v2.trainer.preprocessing.steps.column_filters import (
    LookbehindCombinationColFilter,
    RegexColumnBlacklist,
    TemporalColumnFilter,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


def test_lookbehind_combination_filter():
    df = str_to_pl_df(
        """pred_age,pred_age_within_2_days,pred_age_within_3_days,pred_diagnosis_within_4_days
        3,4,3,2
        3,4,3,3
        4,3,4,1
        """,
    )

    from psycop.common.model_training_v2.loggers.terminal_logger import TerminalLogger

    logger = TerminalLogger()
    result = LookbehindCombinationColFilter(
        lookbehinds={2, 3},
        pred_col_prefix="pred_",
        logger=logger,
    ).apply(df)

    assert len(result.columns) == 3


def test_regex_column_blacklist():
    df = str_to_pl_df(
        """pred_1, test_pred_1
        1, 2""",
    )

    filtered = RegexColumnBlacklist("pred_.*").apply(df)
    assert filtered.columns == [c for c in df.columns if not c.startswith("pred_")]


def test_temporal_column_filter():
    df = str_to_pl_df(
        """timestamp,pred_1,
                      2020-01-01,1,""",
    )
    filtered = TemporalColumnFilter().apply(df)
    assert filtered.columns == ["pred_1"]

import polars as pl

from psycop.common.test_utils.str_to_df import str_to_pl_df
from psycop.projects.t2d.paper_outputs.model_permutation.boolean_features import (
    convert_predictors_to_boolean,
)


def test_convert_predictors_to_boolean():
    input_df = str_to_pl_df(
        """pred_test1,pred_test2,eval_test1,prediction_time_uuid,outc_1,
1,2,1,1,1
2,4,2,2,2
3,6,3,3,3
NaN,NaN,NaN,NaN,NaN"""
    ).lazy()

    boolean_df = convert_predictors_to_boolean(df=input_df, predictor_prefix="pred_").collect()

    for expected_outcome_col in (
        "eval_test1",
        "prediction_time_uuid",
        "outc_1",
        "pred_test1",
        "pred_test2",
    ):
        assert expected_outcome_col in boolean_df.columns
        if "pred_" in expected_outcome_col:
            assert boolean_df[expected_outcome_col].dtype == pl.Int32

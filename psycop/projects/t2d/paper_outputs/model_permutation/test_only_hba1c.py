from psycop.common.test_utils.str_to_df import str_to_pl_df
from psycop.projects.t2d.paper_outputs.model_permutation.only_hba1c import Hba1cOnly


def test_keep_only_hba1c_predictors():
    input_df = str_to_pl_df(
        """pred_test1,pred_hba1c_mean_within_365_days,eval_test1,prediction_time_uuid,outc_1,pred_hba1c_median_within_365_days,
1,2,1,1,1,1
2,4,2,2,2,1
3,6,3,3,3,1
NaN,NaN,NaN,NaN,NaN,NaN"""
    ).lazy()

    hba1c_only_df = (
        Hba1cOnly(lookbehind="365", aggregation_method="mean")
        ._keep_only_hba1c_predictors(  # type: ignore
            df=input_df, predictor_prefix="pred_"
        )
        .collect()
    )

    assert "pred_test1" not in hba1c_only_df.columns
    assert "pred_hba1c_mean_within_365_days" in hba1c_only_df.columns

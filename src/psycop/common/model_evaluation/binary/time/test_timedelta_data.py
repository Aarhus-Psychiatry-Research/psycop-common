from psycop.common.model_evaluation.binary.time.timedelta_data import (
    create_sensitivity_by_time_to_outcome_df,
    get_sensitivity_by_timedelta_df,
    get_timedelta_series,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.common.test_utils.str_to_df import str_to_df


def test_get_timedelta_series():
    input_df = str_to_df(
        """t1_timestamp,t2_timestamp,
    2020-01-01,2020-01-02,
    2020-01-01,2020-01-03,""",
    )

    t2_minus_t1_days = get_timedelta_series(
        direction="t2-t1",
        bin_unit="D",
        df=input_df,
        t2_col_name="t2_timestamp",
        t1_col_name="t1_timestamp",
    )

    assert t2_minus_t1_days.tolist() == [1, 2]

    t1_minus_t2_days = get_timedelta_series(
        direction="t1-t2",
        bin_unit="D",
        df=input_df,
        t2_col_name="t2_timestamp",
        t1_col_name="t1_timestamp",
    )

    assert t1_minus_t2_days.tolist() == [-1, -2]


def test_sensitivity_by_timedelta():
    df = str_to_df(
        """t1_timestamp,t2_timestamp,y,y_hat_probs,
    2020-01-01,2020-01-02,1,1,
    2020-01-01,2020-01-02,1,1,
    2020-01-01,2020-01-02,0,0,
    2020-01-01,2020-01-02,0,0,
    2020-01-01,2020-01-02,0,0,""",
    )

    sensitivity_df = get_sensitivity_by_timedelta_df(
        y=df["y"],
        y_hat_probs=df["y_hat_probs"],
        time_one=df["t1_timestamp"],
        time_two=df["t2_timestamp"],
        direction="t2-t1",
        bins=[0, 10],
        bin_unit="D",
    )

    ci_lower = sensitivity_df["ci_lower"][0]
    assert ci_lower == 1.0


def test_get_sens_by_time_to_outcome_df(synth_eval_dataset: EvalDataset):
    create_sensitivity_by_time_to_outcome_df(
        eval_dataset=synth_eval_dataset,
        outcome_timestamps=synth_eval_dataset.outcome_timestamps,
        prediction_timestamps=synth_eval_dataset.pred_timestamps,
        desired_positive_rate=0.5,
    )

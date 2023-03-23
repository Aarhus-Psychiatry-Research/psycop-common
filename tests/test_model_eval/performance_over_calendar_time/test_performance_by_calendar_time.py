import pandas as pd
from psycop_model_training.model_eval.base_artifacts.plots.performance_over_time import (
    create_performance_by_timedelta,
)
from psycop_model_training.model_eval.dataclasses import EvalDataset
from sklearn.metrics import recall_score


def test_create_performance_by_time_from_event_df(synth_eval_dataset: EvalDataset):
    eval_ds = synth_eval_dataset

    timestamp_df = pd.DataFrame(
        {
            "pred_timestamps": eval_ds.pred_timestamps,
            "outcome_timestamps": eval_ds.outcome_timestamps,
        },
    )
    timestamp_df["differences"] = (
        timestamp_df["outcome_timestamps"] - timestamp_df["pred_timestamps"]
    )

    pass

    df = create_performance_by_timedelta(
        labels=eval_ds.y,
        time_one=eval_ds.outcome_timestamps,
        time_two=eval_ds.pred_timestamps,
        metric_fn=recall_score,
        direction="t2-t1",
        bins=[-500, -100, 0, 100, 500, 1000],
        bin_unit="D",
        bin_continuous_input=True,
        drop_na_events=False,
        min_n_in_bin=5,
    )

    assert df.shape[0] == 6
    assert df["n_in_bin"].sum() >= 4800
    assert df["n_in_bin"].sum() <= 5200
    assert df["metric"].mean() >= 0.04
    assert df["metric"].mean() <= 0.06

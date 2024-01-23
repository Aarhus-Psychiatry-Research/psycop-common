from pathlib import Path

import pandas as pd

from psycop.common.model_training.training_output.artifact_saver.to_disk import ArtifactsToDiskSaver
from psycop.common.model_training.training_output.dataclasses import EvalDataset


def test_eval_dataset_to_disk_series(tmp_path: Path):
    ids = pd.Series([1, 1, 2, 2])
    pred_time_uuids = pd.Series(["1-2021-01-01", "1-2021-01-02", "2-2021-01-01", "2-2021-01-02"])
    pred_timestamps = pd.Series(["2021-01-01", "2021-01-02", "2021-01-01", "2021-01-02"])
    y = pd.Series([0, 1, 1, 0], name="y")
    y_hat_probs = pd.Series([0, 0.2, 0.5, 0], name="y_hat_probs")

    eval_ds = EvalDataset(
        ids=ids,
        pred_time_uuids=pred_time_uuids,
        pred_timestamps=pred_timestamps,
        y=y,
        y_hat_probs=y_hat_probs,
    )

    ArtifactsToDiskSaver.eval_dataset_to_disk(
        eval_dataset=eval_ds, file_path=tmp_path.with_suffix(".parquet")
    )

    eval_df = pd.read_parquet(tmp_path.with_suffix(".parquet"))

    pd.testing.assert_series_equal(y, eval_df.y)
    pd.testing.assert_series_equal(y_hat_probs, eval_df.y_hat_probs)


def test_eval_dataset_to_disk_df(tmp_path: Path):
    ids = pd.Series([1, 1, 2, 2])
    pred_time_uuids = pd.Series(["1-2021-01-01", "1-2021-01-02", "2-2021-01-01", "2-2021-01-02"])
    pred_timestamps = pd.Series(["2021-01-01", "2021-01-02", "2021-01-01", "2021-01-02"])
    y = pd.DataFrame({"y_1": [0, 1, 1, 0], "y_2": [1, 0, 0, 1]})
    y_hat_probs = pd.DataFrame(
        {"y_hat_probs_1": [0, 0.2, 0.5, 0], "y_hat_probs_2": [0.5, 0, 0, 0.9]}
    )

    eval_ds = EvalDataset(
        ids=ids,
        pred_time_uuids=pred_time_uuids,
        pred_timestamps=pred_timestamps,
        y=y,
        y_hat_probs=y_hat_probs,
    )

    ArtifactsToDiskSaver.eval_dataset_to_disk(
        eval_dataset=eval_ds, file_path=tmp_path.with_suffix(".parquet")
    )

    eval_df = pd.read_parquet(tmp_path.with_suffix(".parquet"))

    pd.testing.assert_series_equal(y.y_1, eval_df.y_1)
    pd.testing.assert_series_equal(y.y_2, eval_df.y_2)
    pd.testing.assert_series_equal(y_hat_probs.y_hat_probs_1, eval_df.y_hat_probs_1)
    pd.testing.assert_series_equal(y_hat_probs.y_hat_probs_2, eval_df.y_hat_probs_2)

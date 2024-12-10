from dataclasses import dataclass
from pathlib import Path

import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import PsycopMlflowRun
from psycop.common.types.validated_frame import ValidatedFrame
from psycop.common.types.validator_rules import ColumnExistsRule, ColumnTypeRule


@dataclass(frozen=True)
class MinimalEvalDataset(ValidatedFrame[pl.DataFrame]):
    pred_time_uuid_col_name: str
    pred_prob_col_name: str
    y_col_name: str

    pred_proba_col_rules = (ColumnExistsRule(), ColumnTypeRule(expected_type=pl.Float64))


def minimal_eval_dataset_from_path(path: Path) -> MinimalEvalDataset:
    df = pl.read_parquet(path)
    return MinimalEvalDataset(
        frame=df,
        pred_time_uuid_col_name="pred_time_uuid",
        pred_prob_col_name="y_hat_prob",
        y_col_name="y",
        allow_extra_columns=False,
    )


def minimal_eval_dataset_from_mlflow_run(run: PsycopMlflowRun) -> MinimalEvalDataset:
    path = run.download_artifact(artifact_name="eval_df.parquet")
    return minimal_eval_dataset_from_path(path)

# TODO fh: check this script, evt sammen med E:\frihae\psycop-common\psycop\projects\scz_bp\evaluation\scz_bp_run_evaluation_suite.py



from typing import NewType

import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame
from psycop.common.model_evaluation.binary.time.timedelta_data import get_auroc_by_timedelta_df
from psycop.projects.ect.model_evaluation.uuid_parsers import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
)

TimeFromFirstVisitDF = NewType("TimeFromFirstVisitDF", pl.DataFrame)


def auroc_by_time_from_first_visit_model(
    eval_frame: EvalFrame, all_visits_df: pl.DataFrame
) -> TimeFromFirstVisitDF:
    eval_dataset = parse_dw_ek_borger_from_uuid(parse_timestamp_from_uuid(eval_frame.frame))

    first_visit = (
        all_visits_df.sort("timestamp", descending=False)
        .groupby("dw_ek_borger")
        .head(1)
        .rename({"timestamp": "first_visit_timestamp"})
    )

    joined_df = eval_dataset.join(
        first_visit.select(["first_visit_timestamp", "dw_ek_borger"]), on="dw_ek_borger", how="left"
    ).to_pandas()

    result_df = get_auroc_by_timedelta_df(
        y=joined_df["y"],
        y_hat_probs=joined_df["y_hat_prob"],
        time_one=joined_df["first_visit_timestamp"],
        time_two=joined_df["timestamp"],
        direction="t2-t1",
        bin_unit="M",
        bins=range(0, 60, 6),
        stratified=True,
    )

    return TimeFromFirstVisitDF(pl.from_pandas(result_df))


if __name__ == "__main__":
    import polars as pl

    from psycop.common.feature_generation.loaders.raw.load_visits import (
        physical_visits_to_psychiatry,
    )
    from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame
    from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk

    experiment = "ECT-hparam-structured_only-xgboost-no-lookbehind-filter"
    experiment_path = f"E:/shared_resources/ect/eval_runs/{experiment}_best_run_evaluated_on_test"
    experiment_df = read_eval_df_from_disk(experiment_path)
    eval_frame = EvalFrame(frame=experiment_df, allow_extra_columns=True)
    all_visits_df = pl.from_pandas(physical_visits_to_psychiatry())

    auroc_by_time_from_first_visit_model(eval_frame, all_visits_df)

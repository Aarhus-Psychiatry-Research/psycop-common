import pandas as pd
import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.procedure_codes import (
    get_ect_procedures,
)
from psycop.projects.ect.model_evaluation.uuid_parsers import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
)


def get_first_ect_indicator() -> pd.DataFrame:
    procedure_codes = get_ect_procedures().rename({"procedurekodetekst": "cause"})
    first_ect = procedure_codes.sort("timestamp").groupby("dw_ek_borger").first()

    return first_ect.select(["dw_ek_borger", "timestamp", "cause"]).to_pandas()


def add_first_ect_time_after_prediction_time(prediction_time_df: pl.DataFrame) -> pl.DataFrame:
    procedure_codes = (
        get_ect_procedures()
        .select("dw_ek_borger", "timestamp")
        .rename({"timestamp": "timestamp_outcome"})
    )

    prediction_time_df = parse_timestamp_from_uuid(parse_dw_ek_borger_from_uuid(prediction_time_df))

    only_prediction_times_with_ect = prediction_time_df.join(
        procedure_codes, on="dw_ek_borger", how="inner"
    )
    ect_after_prediction_time = only_prediction_times_with_ect.filter(
        pl.col("timestamp") < pl.col("timestamp_outcome")
    )

    ect_closest_to_prediction_time = ect_after_prediction_time.groupby("pred_time_uuid").agg(
        pl.min("timestamp_outcome")
    )

    result_df = prediction_time_df.join(
        ect_closest_to_prediction_time, on="pred_time_uuid", how="left"
    )

    return result_df


if __name__ == "__main__":
    eval_frame = (
        MlflowClientWrapper()
        .get_run(
            experiment_name="ECT hparam, structured_only, xgboost, no lookbehind filter",
            run_name="inquisitive-koi-243",
        )
        .eval_frame()
    )
    df_vector = add_first_ect_time_after_prediction_time(eval_frame.frame)
    df_vector.filter(pl.col("timestamp_outcome").is_not_null())

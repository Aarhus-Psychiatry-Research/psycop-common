import pandas as pd
import polars as pl

from psycop.common.cross_experiments.cross_project_catalogue import ModelCatalogue
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits
from psycop.common.model_evaluation.utils import bin_continuous_data
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.common.model_training_v2.trainer.preprocessing.steps.geographical_split._geographical_split import (
    add_shak_to_region_mapping,
    load_shak_to_location_mapping,
)
from psycop.projects.restraint.feature_generation.modules.loaders.load_restraint_prediction_timestamps import (
    load_restraint_prediction_timestamps,
)


# change to class subset with methods a la get_preprocessed_dfs (for each individual)
def get_eval_dfs(catalogue: ModelCatalogue) -> pd.DataFrame:
    eval_dfs = catalogue.get_eval_dfs()
    pprs = catalogue.get_predicted_positive_rates()

    # MAKE DFS OPTIONAL
    cvd = eval_dfs["CVD"]
    cvd["y_hat"] = get_predictions_for_positive_rate(pprs["CVD"], cvd["y_hat_prob"])[0]
    cvd["dw_ek_borger"] = cvd["pred_time_uuid"].str.split("-").str[0].astype("int64")
    cvd["timestamp"] = (
        cvd["pred_time_uuid"]
        .str.split("-")
        .apply(lambda x: "-".join(x[1:]))
        .pipe(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
    )
    cvd["model"] = "Cardiovascular disease"

    ect = eval_dfs["ECT"]
    ect["y_hat"] = get_predictions_for_positive_rate(pprs["ECT"], ect["y_hat_prob"])[0]
    ect["dw_ek_borger"] = ect["pred_time_uuid"].str.split("-").str[0].astype("int64")
    ect["timestamp"] = (
        ect["pred_time_uuid"]
        .str.split("-")
        .apply(lambda x: "-".join(x[1:]))
        .pipe(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
    )
    ect["model"] = "Electroconvulsive therapy"

    restraint = eval_dfs["Restraint"]
    restraint["y_hat"] = get_predictions_for_positive_rate(
        pprs["Restraint"], restraint["y_hat_prob"]
    )[0]
    restraint["dw_ek_borger"] = restraint["pred_time_uuid"].str.split("-").str[0].astype("int64")
    restraint["timestamp"] = (
        restraint["pred_time_uuid"]
        .str.split("-")
        .apply(lambda x: "-".join(x[1:]))
        .pipe(pd.to_datetime, format="%Y-%m-%d-%H-%M-%S")
    )
    restraint["model"] = "Physical restraint"
    pred_times = pl.DataFrame(load_restraint_prediction_timestamps()).select(
        pl.col(["dw_ek_borger", "timestamp", "timestamp_discharge"])
    )

    restraint = (
        pl.DataFrame(restraint)
        .with_columns(pl.col("timestamp").dt.cast_time_unit("ns"))
        .join(pred_times, on=["dw_ek_borger", "timestamp"], how="left")
    )

    sczbp = eval_dfs["SCZ_BP"]
    sczbp["y_hat"] = get_predictions_for_positive_rate(pprs["SCZ_BP"], sczbp["y_hat_prob"])[0]
    sczbp["dw_ek_borger"] = sczbp["pred_time_uuid"].str.split("-").str[0].astype("int64")
    sczbp["timestamp"] = (
        sczbp["pred_time_uuid"]
        .str.split("-")
        .apply(lambda x: "-".join(x[1:]))
        .pipe(pd.to_datetime, format="%Y-%m-%d-%H-%M-%S")
    )
    sczbp["model"] = "Schizophrenia or bipolar disorder"

    fai = eval_dfs["FAI"]
    fai["y_hat"] = get_predictions_for_positive_rate(pprs["FAI"], fai["y_hat_prob"])[0]
    fai["dw_ek_borger"] = fai["pred_time_uuid"].str.split("-").str[0].astype("int64")
    fai["timestamp"] = (
        fai["pred_time_uuid"]
        .str.split("-")
        .apply(lambda x: "-".join(x[1:]))
        .pipe(pd.to_datetime, format="%Y-%m-%d-%H-%M-%S")
    )
    fai["model"] = "Involuntary hospitalisation"

    t2d = pd.DataFrame(
        {
            "y": eval_dfs["T2D"]["y"],
            "y_hat_prob": eval_dfs["T2D"]["y_hat_probs"],
            "pred_time_uuid": eval_dfs["T2D"]["pred_time_uuids"],
            "y_hat": get_predictions_for_positive_rate(pprs["T2D"], eval_dfs["T2D"]["y_hat_probs"])[
                0
            ],
            "dw_ek_borger": eval_dfs["T2D"]["ids"],
            "timestamp": eval_dfs["T2D"]["pred_timestamps"],
            "model": "Type 2 diabetes",
        }
    )

    eval_df_cvd_t2d = pd.concat([cvd, t2d, ect])

    shak_to_location_df = load_shak_to_location_mapping()

    visits_start = pl.from_pandas(
        physical_visits(shak_code=6600, timestamp_for_output="start", return_shak_location=True)
    )
    visits_end = pl.from_pandas(physical_visits(shak_code=6600, return_shak_location=True))

    sorted_all_visits_start_df = (
        add_shak_to_region_mapping(
            visits=visits_start,
            shak_to_location_df=shak_to_location_df,
            shak_codes_to_drop=[],
            columns_to_keep=["dw_ek_borger", "timestamp", "unit", "region"],
        )
        .sort(["dw_ek_borger", "timestamp"])
        .to_pandas()
    )

    eval_df_cvd_t2d = eval_df_cvd_t2d.merge(
        sorted_all_visits_start_df, on=["dw_ek_borger", "timestamp"], how="left"
    )

    sorted_all_visits_start_df["timestamp_minus_day"] = (
        sorted_all_visits_start_df.timestamp - pd.Timedelta(days=1)
    )
    eval_df_sczbp = sczbp.merge(
        sorted_all_visits_start_df.drop(columns="timestamp"),
        left_on=["dw_ek_borger", "timestamp"],
        right_on=["dw_ek_borger", "timestamp_minus_day"],
        how="left",
    )

    sorted_all_visits_end_df = add_shak_to_region_mapping(
        visits=visits_end,
        shak_to_location_df=shak_to_location_df,
        shak_codes_to_drop=[],
        columns_to_keep=["dw_ek_borger", "timestamp", "unit", "region"],
    ).sort(["dw_ek_borger", "timestamp"])

    eval_df_restraint = (
        restraint.join(
            sorted_all_visits_end_df,
            left_on=["dw_ek_borger", "timestamp_discharge"],
            right_on=["dw_ek_borger", "timestamp"],
            how="left",
        )
        .to_pandas()
        .drop(columns="timestamp_discharge")
    )

    eval_df_fai = (
        pl.DataFrame(fai)
        .join(sorted_all_visits_end_df, on=["dw_ek_borger", "timestamp"], how="left")
        .to_pandas()
    )

    eval_df = pd.concat([eval_df_cvd_t2d, eval_df_restraint, eval_df_fai, eval_df_sczbp])

    # ECT  >7 days (to not include patients admitted for planned ECT) and ≤67 days

    eval_df = eval_df.merge(birthdays(), on="dw_ek_borger", how="left")
    eval_df["age"] = (eval_df["timestamp"] - eval_df["date_of_birth"]).dt.total_seconds() / (
        60 * 60 * 24
    )
    eval_df["age"] = (eval_df["age"] / 365.25).astype("int")
    eval_df["age_group"] = bin_continuous_data(series=eval_df["age"], bins=[18, 25, 40, 55, 70])[0]
    eval_df = eval_df[eval_df["age_group"].notna()]

    eval_df = eval_df.merge(sex_female(), on="dw_ek_borger", how="left")
    eval_df["sex"] = eval_df["sex_female"].replace({True: "Female", False: "Male"})

    return eval_df.drop(columns=["pred_time_uuid", "date_of_birth", "sex_female"])


if __name__ == "__main__":
    eval_df = get_eval_dfs(
        ModelCatalogue(projects=["CVD", "ECT", "FAI", "Restraint", "SCZ_BP", "T2D"])
    )

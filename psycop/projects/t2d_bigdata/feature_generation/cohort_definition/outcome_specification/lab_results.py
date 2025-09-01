import pandas as pd

from psycop.common.feature_generation.loaders.raw.load_lab_results import (
    fasting_p_glc,
    hba1c,
    ogtt,
    unscheduled_p_glc,
)


def get_rows_above_value(
    value: float, df: pd.DataFrame, value_type: str
) -> pd.DataFrame:  # TODO: new loaders?
    output_df = df[df["value"] > value]
    output_df["value_type"] = value_type

    return output_df


def get_hba1cs_above_threshold() -> pd.DataFrame:
    return get_rows_above_value(df=hba1c(), value=47.9, value_type="hba1c")


def get_unscheduled_p_glc_above_threshold() -> pd.DataFrame:
    return get_rows_above_value(df=unscheduled_p_glc(), value=11.0, value_type="unscheduled_p_glc")


def get_ogtt_above_threshold() -> pd.DataFrame:
    return get_rows_above_value(df=ogtt(), value=11.0, value_type="ogtt")


def get_fasting_glc_above_threshold() -> pd.DataFrame:
    return get_rows_above_value(df=fasting_p_glc(), value=6.9, value_type="fasting_p_glc")


def get_diabetes_lab_results_above_threshold() -> pd.DataFrame:
    hba1cs = get_hba1cs_above_threshold()
    unscheduled_p_glc = get_unscheduled_p_glc_above_threshold()
    fasting_glc = get_fasting_glc_above_threshold()
    ogtt = get_ogtt_above_threshold()

    return pd.concat([hba1cs, unscheduled_p_glc, fasting_glc, ogtt], axis=0)


def get_first_diabetes_lab_result_above_threshold() -> pd.DataFrame:
    df = get_diabetes_lab_results_above_threshold()

    first_lab_result_above_threshold = (
        df.sort_values("timestamp").groupby("dw_ek_borger").first().reset_index(drop=False)
    )

    first_lab_result_above_threshold["value"] = 1
    return first_lab_result_above_threshold[["dw_ek_borger", "timestamp", "value"]]


if __name__ == "__main__":
    df = get_first_diabetes_lab_result_above_threshold()

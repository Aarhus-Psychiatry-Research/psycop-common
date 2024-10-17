import pandas as pd


def _get_pred_cols_df(df: pd.DataFrame) -> pd.DataFrame:
    """Get columns that start with pred_."""
    pred_cols = [col for col in df.columns if col.startswith("pred_")]  # type: ignore

    return df[pred_cols]


def _generate_general_descriptive_stats(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Generate descriptive stats table."""

    unique_patients = df["dw_ek_borger"].nunique()
    mean_entries_per_patient = df.shape[0] / unique_patients
    mean_age = df["pred_age_in_years"].mean()

    pred_cols = _get_pred_cols_df(df)
    na_ratios = pred_cols.isna().sum().sum() / (pred_cols.shape[0] * pred_cols.shape[1])

    outcomes = df["outc_AcuteAdmission_within_30_days_maximum_fallback_0_dichotomous"].sum()
    outcome_rate = outcomes / df.shape[0]

    unique_patients_with_outcomes = df[
        df["outc_AcuteAdmission_within_30_days_maximum_fallback_0_dichotomous"] == 1
    ]["dw_ek_borger"].nunique()

    print(
        df[df["outc_AcuteAdmission_within_30_days_maximum_fallback_0_dichotomous"] == 1][
            "dw_ek_borger"
        ]
        .value_counts()
        .head(10)
    )

    stats = pd.DataFrame(
        data={
            "split_name": [split_name],
            "number_of_entries": [df.shape[0]],
            "unique_patients": [unique_patients],
            "mean_entries_per_patient": [mean_entries_per_patient],
            "mean_age": [mean_age],
            "outcome_rate": [outcome_rate],
            "outcomes": [outcomes],
            "unique_patients_with_outcomes": [unique_patients_with_outcomes],
            "na_ratios": [na_ratios],
        }
    )

    return stats


def main():
    df_train = pd.read_parquet(
        "E:/shared_resources/AcuteSomaticAdmission/feature_set/flattened_datasets/somatic_tot_experiments/train.parquet"
    )
    df_val = pd.read_parquet(
        "E:/shared_resources/AcuteSomaticAdmission/feature_set/flattened_datasets/somatic_tot_experiments/val.parquet"
    )
    df_test = pd.read_parquet(
        "E:/shared_resources/AcuteSomaticAdmission/feature_set/flattened_datasets/somatic_tot_experiments/test.parquet"
    )

    train_stats = _generate_general_descriptive_stats(df_train, split_name="train")
    val_stats = _generate_general_descriptive_stats(df_val, split_name="val")
    test_stats = _generate_general_descriptive_stats(df_test, split_name="test")

    df_tot = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)
    tot_stats = _generate_general_descriptive_stats(df_tot, split_name="tot")

    stats = pd.concat([train_stats, val_stats, test_stats], axis=0).reset_index(drop=True)

"""Tables for description and evaluation of models and patient population."""
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import roc_auc_score

from psycop_model_training.model_eval.dataclasses import EvalDataset
from psycop_model_training.utils.utils import bin_continuous_data

def _calc_auc_and_n(
    df: pd.DataFrame,
    pred_probs_col_name: str,
    outcome_col_name: str,
) -> pd.Series:
    """Calculate auc and number of data points per group.

    Args:
        df (pd.DataFrame): DataFrame containing predicted probabilities and labels
        pred_probs_col_name (str): The column containing predicted probabilities
        outcome_col_name (str): THe containing the labels

    Returns:
        pd.Series: Series containing AUC and N
    """
    auc = roc_auc_score(df[outcome_col_name], df[pred_probs_col_name])
    n = len(df)
    return pd.Series([auc, n], index=["AUC", "N"])


def output_table(
    output_format: str,
    df: pd.DataFrame,
) -> Union[pd.DataFrame, wandb.Table]:
    """Output table in specified format."""
    if output_format == "html":
        return df.reset_index(drop=True).to_html()
    elif output_format == "df":
        return df.reset_index(drop=True)
    elif output_format == "wandb_table":
        return wandb.Table(dataframe=df)
    else:
        raise ValueError("Output format does not match anything that is allowed")

def _table_1_default_df():
    """Create default table 1 dataframe.
    Returns:
        pd.DataFrame: Default table 1 dataframe. Includes columns for category, two statistics and there units."""
    return pd.DataFrame(
        columns=["category", "stat_1", "stat_1_unit", "stat_2", "stat_2_unit"],
    )

def _generate_age_stats(
    eval_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Add age stats to table 1."""
    
    df = _table_1_default_df()
    
    age_mean = np.round(eval_dataset["age"].mean(), 2)

    age_span = f'{eval_dataset["age"].quantile(0.05)} - {eval_dataset["age"].quantile(0.95)}'

    df = df.append(
        {
            "category": "(visit_level) age (mean / interval)",
            "stat_1": age_mean,
            "stat_1_unit": "years",
            "stat_2": age_span,
            "stat_2_unit": "years",
        },
        ignore_index=True,
    )
    age_counts = bin_continuous_data(eval_dataset["age"], bins=[0, 18, 35, 60, 100]).value_counts()

    age_percentages = np.round(age_counts / len(eval_dataset) * 100, 2)

    for i, _ in enumerate(age_counts):
        df = df.append(
            {
                "category": f"(visit level) age {age_counts.index[i]}",
                "stat_1": int(age_counts.iloc[i]),
                "stat_1_unit": "patients",
                "stat_2": age_percentages.iloc[i],
                "stat_2_unit": "%",
            },
            ignore_index=True,
        )

    return df


def _generate_sex_stats(
    eval_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Add sex stats to table 1."""
    
    df = _table_1_default_df()

    sex_counts = eval_dataset["sex"].value_counts()
    sex_percentages = sex_counts / len(eval_dataset) * 100

    for i, n in enumerate(sex_counts):
        if n < 5:
            warnings.warn(
                "WARNING: One of the sex categories has less than 5 individuals. This category will be excluded from the table.",
            )
            return df

        df = df.append(
            {
                "category": f"(visit level) {sex_counts.index[i]}",
                "stat_1": int(sex_counts[i]),
                "stat_1_unit": "patients",
                "stat_2": sex_percentages[i],
                "stat_2_unit": "%",
            },
            ignore_index=True,
        )

    return df


def _generate_eval_col_stats(eval_dataset: pd.DataFrame) -> pd.DataFrame:
    """Generate stats for all eval_ columns to table 1.

    Finds all columns starting with 'eval_' and adds visit level stats
    for these columns. Checks if the column is binary or continuous and
    adds stats accordingly.
    """

    df = _table_1_default_df()

    eval_cols = [col for col in eval_dataset.columns if col.startswith("eval_")]

    for col in eval_cols:
        if len(eval_dataset[col].unique()) == 2:
            # Binary variable stats:
            col_count = eval_dataset[col].value_counts()
            col_percentage = col_count / len(eval_dataset) * 100

            if col_count[0] < 5 or col_count[1] < 5:
                warnings.warn(
                    f"WARNING: One of categories in {col} has less than 5 individuals. This category will be excluded from the table.",
                )
            else:
                df = df.append(
                    {
                        "category": f"(visit level) {col} ",
                        "stat_1": int(col_count[1]),
                        "stat_1_unit": "patients",
                        "stat_2": col_percentage[1],
                        "stat_2_unit": "%",
                    },
                    ignore_index=True,
                )

        elif len(eval_dataset[col].unique()) > 2:
            # Continuous variable stats:
            col_mean = np.round(eval_dataset[col].mean(), 2)
            col_std = np.round(eval_dataset[col].std(), 2)
            df = df.append(
                {
                    "category": f"(visit level) {col}",
                    "stat_1": col_mean,
                    "stat_1_unit": "mean",
                    "stat_2": col_std,
                    "stat_2_unit": "std",
                },
                ignore_index=True,
            )

        else:
            warnings.warn(
                f"WARNING: {col} has only one value. This column will be excluded from the table.",
            )
            
    return df


def _generate_visit_level_stats(
    eval_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Generate all visit level stats to table 1."""
    # Stats for eval_ cols
    df = _generate_eval_col_stats(eval_dataset)

    # General stats
    visits_followed_by_positive_outcome = int(eval_dataset["y"].sum())
    visits_followed_by_positive_outcome_percentage = np.round((
        visits_followed_by_positive_outcome / len(eval_dataset) * 100
    ), 2)

    df = df.append(
        {
            "category": "(visit_level) visits followed by positive outcome",
            "stat_1": visits_followed_by_positive_outcome,
            "stat_1_unit": "visits",
            "stat_2": visits_followed_by_positive_outcome_percentage,
            "stat_2_unit": "%",
        },
        ignore_index=True,
    )

    return df


def _calc_time_to_first_positive_outcome_stats(
    patients_with_positive_outcome_data: pd.DataFrame,
) -> float:
    """Calculate mean time to first positive outcome (currently very slow)."""

    grouped_data = patients_with_positive_outcome_data.groupby("ids")

    time_to_first_positive_outcome = grouped_data.apply(lambda x: x["outcome_timestamps"].min() - x["pred_timestamps"].min())

    # Convert to days (float)
    time_to_first_positive_outcome = time_to_first_positive_outcome.dt.total_seconds() / (24 * 60 * 60)

    return np.round(time_to_first_positive_outcome.mean(), 2), np.round(
        time_to_first_positive_outcome.std(),
        2,
    )


def _generate_patient_level_stats(
    eval_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Add patient level stats to table 1."""
    
    df = _table_1_default_df()

    # General stats
    patients_with_positive_outcome = eval_dataset[eval_dataset["y"] == 1][
        "ids"
    ].unique()
    n_patients_with_positive_outcome = len(patients_with_positive_outcome)
    patients_with_positive_outcome_percentage = np.round((
        n_patients_with_positive_outcome / len(eval_dataset["ids"].unique()) * 100
    ), 2)

    df = df.append(
        {
            "category": "(patient_level) patients_with_positive_outcome",
            "stat_1": n_patients_with_positive_outcome,
            "stat_1_unit": "visits",
            "stat_2": patients_with_positive_outcome_percentage,
            "stat_2_unit": "%",
        },
        ignore_index=True,
    )

    patients_with_positive_outcome_data = eval_dataset[
        eval_dataset["ids"].isin(patients_with_positive_outcome)
    ]

    (
        mean_time_to_first_positive_outcome, 
        std_time_to_first_positive_outomce
    ) = _calc_time_to_first_positive_outcome_stats(patients_with_positive_outcome_data)

    df = df.append(
        {
            "category": "(patient level) time_to_first_positive_outcome",
            "stat_1": mean_time_to_first_positive_outcome,
            "stat_1_unit": "mean",
            "stat_2": std_time_to_first_positive_outomce,
            "stat_2_unit": "std",
        },
        ignore_index=True,
    )

    return df


def generate_table_1(
    eval_dataset: EvalDataset,
    output_format: str = "df",
    save_path: Optional[Path] = None,
) -> Union[pd.DataFrame, wandb.Table]:
    """Generate table 1. Calculates relevant statistics from the evaluation
    dataset and returns a pandas dataframe or wandb table. If save_path is
    provided, the table is saved as a csv file.

    Args:
        eval_dataset (EvalDataset): Evaluation dataset.
        output_format (str, optional): Output format. Defaults to "df".
        save_path (Optional[Path], optional): Path to save the table as a csv file. Defaults to None.

    Returns:
        Union[pd.DataFrame, wandb.Table]: Table 1.
    """

    eval_dataset = eval_dataset.to_df()

    if "age" in eval_dataset.columns:
        age_stats = _generate_age_stats(eval_dataset)

    if "sex" in eval_dataset.columns:
        sex_stats = _generate_sex_stats(eval_dataset)

    visit_level_stats = _generate_visit_level_stats(eval_dataset)

    patient_level_stats = _generate_patient_level_stats(eval_dataset)

    table_1_df_list = [age_stats, sex_stats, visit_level_stats, patient_level_stats]
    table_1 = pd.concat(table_1_df_list, ignore_index=True)

    if save_path is not None:
        output_table(output_format="df", df=table_1)

        table_1.to_csv(save_path, index=False)

    return output_table(output_format=output_format, df=table_1)


def generate_feature_importances_table(
    feature_importance_dict: dict[str, float],
    output_format: str = "wandb_table",
) -> Union[pd.DataFrame, wandb.Table]:
    """Generate table with feature importances.

    Args:
        feature_importance_dict (dict[str, float]): Dictionary with feature importances
        output_format (str, optional): The output format. Takes one of "html", "df", "wandb_table". Defaults to "wandb_table".

    Returns:
        Union[pd.DataFrame, wandb.Table]: The table with feature importances
    """
    feature_names = list(feature_importance_dict.keys())
    feature_importances = list(feature_importance_dict.values())

    df = pd.DataFrame(
        {"predictor": feature_names, "feature_importance": feature_importances},
    )
    df = df.sort_values("feature_importance", ascending=False)

    return output_table(output_format=output_format, df=df)


def generate_selected_features_table(
    selected_features_dict: dict[str, bool],
    output_format: str = "wandb_table",
    removed_first: bool = True,
) -> Union[pd.DataFrame, wandb.Table]:
    """Get table with feature selection results.

    Args:
        selected_features_dict (dict[str, bool]): Dictionary with feature selection results
        output_format (str, optional): The output format. Takes one of "html", "df", "wandb_table". Defaults to "wandb_table".
        removed_first (bool, optional): Ordering of features in the table, whether the removed features are first. Defaults to True.
    """

    feature_names = list(selected_features_dict.keys())
    is_selected = list(selected_features_dict.values())

    df = pd.DataFrame(
        {
            "predictor": feature_names,
            "selected": is_selected,
        },
    )

    # Sort df so removed columns appear first
    df = df.sort_values("selected", ascending=removed_first)

    return output_table(output_format=output_format, df=df)

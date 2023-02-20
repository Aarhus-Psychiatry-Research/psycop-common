"""Tables for evaluation of models."""
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import roc_auc_score

from psycop_model_training.model_eval.dataclasses import EvalDataset


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


def generate_table_1(
    eval_dataset: EvalDataset,
    output_format: str = "wandb_table",
) -> Union[pd.DataFrame, wandb.Table]:
    """Generate table 1."""

    eval_dataset = eval_dataset.to_df()

    df = pd.DataFrame(columns=["category", "stat_1", "stat_1_unit", "stat_2", "stat_2_unit"])

    if 'age' in eval_dataset.columns:
        df = _add_age_stats(eval_dataset, df)

    if 'sex' in eval_dataset.columns:
        df = _add_sex_stats(eval_dataset, df)

    df = _add_visit_level_stats(eval_dataset, df)

    return output_table(output_format=output_format, df=df)

def _add_age_stats(
    eval_dataset: pd.DataFrame,
    df : pd.DataFrame) -> pd.DataFrame:
    """Add age stats to table 1."""
    age_mean = np.round(eval_dataset["age"].mean(), 2)

    age_span = f'{eval_dataset["age"].min()} - {eval_dataset["age"].max()}'

    df = df.append({'category': 'age (mean / interval)', 'stat_1': age_mean, 'stat_1_unit': 'years', 'stat_2': age_span, 'stat_2_unit': 'years'}, ignore_index=True)

    age_bins = np.round(np.linspace(eval_dataset["age"].min(), eval_dataset["age"].max(), 5), 0)

    age_counts = pd.cut(eval_dataset["age"], bins=age_bins).value_counts().sort_index()
    age_percentages = age_counts/len(eval_dataset)*100
    
    temp = pd.DataFrame(columns=["category", "stat_1", "stat_1_unit", "stat_2", "stat_2_unit"])
    for i, n in enumerate(age_counts):

        if n < 5:
            warnings.warn("WARNING: One of the age categories has less than 5 individuals. This category will be excluded from the table.")
            return df

        temp = temp.append({'category': f'age {age_counts.index[i]}', 'stat_1': int(age_counts.iloc[i]), 'stat_1_unit': 'patients', 'stat_2': age_percentages.iloc[i], 'stat_2_unit': '%'}, ignore_index=True)

    return df.append(temp)

def _add_sex_stats(
    eval_dataset: pd.DataFrame,
    df: pd.DataFrame) -> pd.DataFrame:
    """Add sex stats to table 1."""
    
    sex_counts = eval_dataset['sex'].value_counts()
    sex_percentages = sex_counts/len(eval_dataset)*100

    temp = pd.DataFrame(columns=["category", "stat_1", "stat_1_unit", "stat_2", "stat_2_unit"])
    for i, n in enumerate(sex_counts):
        if n < 5:
            warnings.warn("WARNING: One of the sex categories has less than 5 individuals. This category will be excluded from the table.")
            return df

        temp = temp.append({'category': sex_counts.index[i], 'stat_1': int(sex_counts[i]), 'stat_1_unit': 'patients', 'stat_2': sex_percentages[i], 'stat_2_unit': '%'}, ignore_index=True)
    
    return df.append(temp)


def _add_visit_level_stats(eval_dataset: pd.DataFrame,
    df: pd.DataFrame) -> pd.DataFrame:
    """Add visit level stats to table 1. Finds all columns starting with 'eval_' and adds stats for these columns. 
    Checks if the column is binary or continuous and adds stats accordingly. """

    eval_cols = [col for col in eval_dataset.columns if col.startswith('eval_')]

    for col in eval_cols:
        if len(eval_dataset[col].unique()) == 2:

            # Binary variable stats:
            col_count = eval_dataset[col].value_counts()
            col_percentage = col_count/len(eval_dataset)*100
            
            if col_count[0] < 5 or col_count[1] < 5:
                    warnings.warn(f"WARNING: One of categories in {col} has less than 5 individuals. This category will be excluded from the table.")
            else:  
                df = df.append({'category': f'{col} ', 'stat_1': int(col_count[1]), 'stat_1_unit': 'patients', 'stat_2': col_percentage[1], 'stat_2_unit': '%'}, ignore_index=True)
        
        elif len(eval_dataset[col].unique()) > 2:
            
            # Continuous variable stats:
            col_mean = np.round(eval_dataset[col].mean(), 2)
            col_std = np.round(eval_dataset[col].std(), 2)
            df = df.append({'category': f'{col} ', 'stat_1': col_mean, 'stat_1_unit': 'mean', 'stat_2': col_std, 'stat_2_unit': 'std'}, ignore_index=True)
        
        else: 
            warnings.warn(f"WARNING: {col} has only one value. This column will be excluded from the table.")
    
    return df
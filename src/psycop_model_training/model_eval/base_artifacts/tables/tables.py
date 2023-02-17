"""Tables for evaluation of models."""
from pathlib import Path
from typing import Optional, Union

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
    age: bool = True,
    sex: bool = True,
    visit_level_cats: Optional[list[str]] = None,
    patient_level_cats: Optional[list[str]] = None,
    output_format: str = "wandb_table",
) -> Union[pd.DataFrame, wandb.Table]:

    df = pd.DataFrame(columns=["category", "stat_1", "stat_1_unit" "stat_2", "stat_2_unit"])

    if age:
        df = _add_age_stats(eval_dataset, df)

    if sex:
        df = _add_sex_stats(eval_dataset, df)
        
    if visit_level_cats is not None:
        for cat in visit_level_cats:

            df = _add_visit_level_stats(eval_dataset, df, cat)

    return output_table(output_format=output_format, df=df)

def _add_age_stats(
    eval_dataset: EvalDataset,
    df : pd.DataFrame) -> pd.DataFrame:
    """Add age stats to table 1."""
    age_median = eval_dataset.age.median()
    age_span = f'{eval_dataset.age.min()} - {eval_dataset.age.max()}'

    df = df.append({'category': 'age', 'stat_1': age_median, 'stat_1_unit': 'years', 'stat_2': age_span, 'stat_2_unit': 'years'}, ignore_index=True)
    return df

def _add_sex_stats(
    eval_dataset: EvalDataset,
    df: pd.DataFrame) -> pd.DataFrame:
    """Add sex stats to table 1."""
    
    sex_counts = eval_dataset.sex.value_counts()

    df = df.append({'category': 'male', 'stat_1': sex_counts[0], 'stat_1_unit': 'patients', 'stat_2': sex_counts[0]/len(eval_dataset)*100, 'stat_2_unit': '%'}, ignore_index=True)
    df = df.append({'category': 'female', 'stat_1': sex_counts[1], 'stat_1_unit': 'patients', 'stat_2': sex_counts[1]/len(eval_dataset)*100, 'stat_2_unit': '%'}, ignore_index=True)
    
    return df

def _add_visit_level_stats(eval_dataset: EvalDataset,
    df: pd.DataFrame,
    cat: str) -> pd.DataFrame:
    """Add visit level stats to table 1."""
    
    col = [eval_dataset[col] for col in eval_dataset.columns if cat in col]
    
    if len(col) > 1:
        raise ValueError(f"Error when generating table 1. More than one column was found in the eval dataset containing {cat}.")
    if len(col) == 0:
        raise ValueError(f"Error when generating table 1. Trying to calculate statistic for {cat} but no matching colum was found in the eval dataset.")

    col = col[0]
    
    col_counts = col.value_counts()
    if len(col_counts) > 2:
        raise ValueError(f"Error when generating table 1. The column {col.name} found for the category {cat} is not dichotomous and therefore statistics could not be calculated.")

    df = df.append({'category': cat, 'stat_1': col_counts[1], 'stat_1_unit': 'patients', 'stat_2': col_counts[1]/len(eval_dataset)*100, 'stat_2_unit': '%'}, ignore_index=True)
    
    return df
"""Utility functions for model training."""
import pandas as pd
from psycop_model_training.config_schemas.data import ColumnNamesSchema
from psycop_model_training.model_eval.dataclasses import EvalDataset


def create_eval_dataset(
    col_names: ColumnNamesSchema,
    outcome_col_name: str,
    df: pd.DataFrame,
):
    """Create an evaluation dataset object from a dataframe and
    ColumnNamesSchema."""
    # Check if custom attribute exists:
    custom_col_names = col_names.custom_columns

    custom_columns = {}

    if custom_col_names is not None:
        custom_columns = {col_name: df[col_name] for col_name in custom_col_names}

    # Add all eval_ columns to custom_columns attribute
    eval_columns = {
        col_name: df[col_name]
        for col_name in df.columns
        if col_name.startswith("eval_")
    }
    custom_columns = (
        custom_columns.update(eval_columns) if len(eval_columns) > 0 else custom_columns
    )

    eval_dataset = EvalDataset(
        ids=df[col_names.id],
        y=df[outcome_col_name],
        y_hat_probs=df["y_hat_prob"],
        y_hat_int=df["y_hat_prob"].round(),
        pred_timestamps=df[col_names.pred_timestamp],
        outcome_timestamps=df[col_names.outcome_timestamp],
        age=df[col_names.age],
        is_female=df[col_names.is_female],
        exclusion_timestamps=df[col_names.exclusion_timestamp]
        if col_names.exclusion_timestamp
        else None,
        custom_columns=custom_columns if len(custom_columns) > 0 else None,
    )

    return eval_dataset

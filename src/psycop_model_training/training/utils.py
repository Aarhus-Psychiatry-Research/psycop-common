import pandas as pd

from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.model_eval.dataclasses import EvalDataset


def create_eval_dataset(cfg: FullConfigSchema, outcome_col_name: str, df: pd.DataFrame):
    """Create an evaluation dataset object from a dataframe and
    FullConfigSchema.
    """
    # Check if custom attribute exists
    if hasattr(cfg.data.col_name, "custom"):
        custom_col_names = cfg.data.col_name.custom
    else:
        custom_col_names = None

    if custom_col_names:
        custom_columns = {col_name: df[col_name] for col_name in custom_col_names}

    eval_dataset = EvalDataset(
        ids=df[cfg.data.col_name.id],
        y=df[outcome_col_name],
        y_hat_probs=df["y_hat_prob"],
        y_hat_int=df["y_hat_prob"].round(),
        pred_timestamps=df[cfg.data.col_name.pred_timestamp],
        outcome_timestamps=df[cfg.data.col_name.outcome_timestamp],
        age=df[cfg.data.col_name.age],
        exclusion_timestamps=df[cfg.data.col_name.exclusion_timestamp],
        custom_columns=custom_columns if custom_col_names else None,
    )

    return eval_dataset

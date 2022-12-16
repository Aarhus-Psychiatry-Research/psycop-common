import pandas as pd

from psycop_model_training.model_eval.dataclasses import EvalDataset
from psycop_model_training.utils.config_schemas import FullConfigSchema


def create_eval_dataset(cfg: FullConfigSchema, outcome_col_name: str, df: pd.DataFrame):
    """Create an evaluation dataset object from a dataframe and
    FullConfigSchema."""

    eval_dataset = EvalDataset(
        ids=df[cfg.data.col_name.id],
        y=df[outcome_col_name],
        y_hat_probs=df["y_hat_prob"],
        y_hat_int=df["y_hat_prob"].round(),
        pred_timestamps=df[cfg.data.col_name.pred_timestamp],
        outcome_timestamps=df[cfg.data.col_name.outcome_timestamp],
        age=df[cfg.data.col_name.age],
        exclusion_timestamps=df[cfg.data.col_name.exclusion_timestamp],
    )

    if cfg.data.col_name.custom:
        if cfg.data.col_name.custom.n_hba1c:
            eval_dataset.custom.n_hba1c = df[cfg.data.col_name.custom.n_hba1c]

    return eval_dataset

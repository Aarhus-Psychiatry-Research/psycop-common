"""Helpers and example code for evaluating an already trained model based on a
pickle file with model predictions and hydra config.

Possible extensions (JIT when needed):
- Load most recent pickle file from 'evaluation_results' folder
- Evaluate all models in 'evaluation_results' folder
- CLI for evaluating a model
"""
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Union

import pandas as pd
from omegaconf.dictconfig import DictConfig

from psycopt2d.utils import PROJECT_ROOT, read_pickle
from psycopt2d.visualization import plot_auc_by_time_from_first_visit


def infer_col_names(
    df: pd.DataFrame,
    prefix: str,
    allow_multiple: bool = True,
) -> Union[str, list[str]]:
    """Infer col names based on prefix."""
    col_name = [c for c in df.columns if c.startswith(prefix)]

    if len(col_name) == 1:
        return col_name[0]
    elif len(col_name) > 1:
        if allow_multiple:
            return col_name
        raise ValueError(
            f"Multipel columns found and allow_multiple is {allow_multiple}.",
        )
    else:
        raise ValueError("More than one outcome inferred")


def infer_outcome_col_name(
    df: pd.DataFrame,
    prefix: str = "outc_",
    allow_multiple: bool = True,
) -> Union[str, list[str]]:
    """Infer the outcome column name from the dataframe."""
    return infer_col_names(df=df, prefix=prefix, allow_multiple=allow_multiple)


def infer_predictor_col_name(
    df: pd.DataFrame,
    prefix: str = "pred_",
    allow_multiple: bool = True,
) -> Union[str, list[str]]:
    """Get the predictors that are used in the model."""
    return infer_col_names(df=df, prefix=prefix, allow_multiple=allow_multiple)


def infer_look_distance(
    col_name: Union[Iterable[str], str],
    regex_pattern: str = r"within_(\d+)_days",
    allow_multiple: bool = True,
) -> list[Union[int, float]]:
    """Infer look distances from col names."""
    # E.g. "outc_within_1_days" = 1
    # E.g. "outc_within_2_days" = 2
    # E.g. "pred_within_3_days" = 3
    # E.g. "pred_within_3_days" = 3

    look_distances: list[Union[int, float]] = []

    if isinstance(col_name, Iterable) and not isinstance(col_name, str):
        for c_name in col_name:
            look_distances += infer_look_distance(
                col_name=c_name,
                regex_pattern=regex_pattern,
            )
    else:
        look_distances = re.findall(pattern=regex_pattern, string=col_name)

    if len(look_distances) > 1 and not allow_multiple:
        raise ValueError(
            f"Multiple col names provided and allow_multiple is {allow_multiple}.",
        )

    return look_distances


def load_model_predictions_and_cfg(path: Path) -> tuple[pd.DataFrame, DictConfig]:
    """Load model predictions and config file from a pickle file.

    Args:
        path (Path): Path to pickle file
    """
    obj = read_pickle(path)
    return obj["df"], obj["cfg"]


if __name__ == "__main__":

    eval_path = PROJECT_ROOT / "evaluation_results"
    eval_df, cfg = load_model_predictions_and_cfg(
        eval_path
        # insert your own model path here
        / "eval_model_name-xgboost_require_imputation-True_args-n_estimators-100_tree_method-auto_2022_09_22_10_52.pkl",
    )

    train_col_names = infer_predictor_col_name(df=eval_df)
    y_col_name = infer_outcome_col_name(df=eval_df)

    Y_HAT_PROB_COL_NAME = "y_hat_prob"  # change to 'y_hat_prob_oof' if using cv

    first_visit_timestamp = eval_df.groupby(cfg.data.id_col_name)[
        cfg.data.pred_timestamp_col_name
    ].transform("min")

    # Do whatever extra evaluation you want to here
    p = plot_auc_by_time_from_first_visit(
        labels=eval_df[y_col_name],
        y_hat_probs=eval_df[Y_HAT_PROB_COL_NAME],
        first_visit_timestamps=first_visit_timestamp,
        prediction_timestamps=eval_df[cfg.data.pred_timestamp_col_name],
    )

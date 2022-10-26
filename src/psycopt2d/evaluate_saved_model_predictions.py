"""Helpers and example code for evaluating an already trained model based on a
pickle file with model predictions and hydra config.

Possible extensions (JIT when needed):
- Load most recent directory from 'evaluation_results'/Overtaci equivalent folder
- Evaluate all models in 'evaluation_results' folder
- CLI for evaluating a model
"""
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Union

import pandas as pd
from omegaconf import DictConfig

from psycopt2d.utils.utils import (
    PROJECT_ROOT,
    infer_outcome_col_name,
    infer_predictor_col_name,
    infer_y_hat_prob_col_name,
    load_evaluation_data,
    read_pickle,
)
from psycopt2d.visualization import plot_auc_by_time_from_first_visit


def infer_look_distance(
    col_name: Union[Iterable[str], str],
    regex_pattern: str = r"within_(\d+)_days",
    allow_multiple: bool = True,
) -> list[str]:
    """Infer look distances from col names."""
    # E.g. "outc_within_1_days" = 1
    # E.g. "outc_within_2_days" = 2
    # E.g. "pred_within_3_days" = 3
    # E.g. "pred_within_3_days" = 3

    look_distances: list[str] = []

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
    # change to whatever modele you wish to evaluate
    eval_data = load_evaluation_data(eval_path / "2022_10_18_13_23_2h3cxref")

    train_col_names = infer_predictor_col_name(df=eval_data.df)
    y_col_name = infer_outcome_col_name(df=eval_data.df)

    y_hat_prob_name = infer_y_hat_prob_col_name(eval_data.df)

    first_visit_timestamp = eval_data.df.groupby(eval_data.cfg.data.id_col_name)[
        eval_data.cfg.data.pred_timestamp_col_name
    ].transform("min")

    # Do whatever extra evaluation you want to here
    p = plot_auc_by_time_from_first_visit(
        first_visit_timestamps=first_visit_timestamp,
    )

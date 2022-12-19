"""Helpers and example code for evaluating an already trained model based on a
pickle file with model predictions and hydra config.

Possible extensions (JIT when needed):
- Load most recent directory from 'evaluation_results'/Overtaci equivalent folder
- Evaluate all models in 'evaluation_results' folder
- CLI for evaluating a model
"""
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from psycop_model_training.model_eval.plots import plot_auc_by_time_from_first_visit
from psycop_model_training.utils.utils import (
    PROJECT_ROOT,
    load_evaluation_data,
    read_pickle,
)
from psycop_model_training.utils.col_name_inference import infer_outcome_col_name, infer_predictor_col_name, \
    infer_y_hat_prob_col_name


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

    first_visit_timestamp = eval_data.df.groupby(eval_data.cfg.data.col_name.id)[
        eval_data.cfg.data.col_name.pred_timestamp
    ].transform("min")

    # Do whatever extra evaluation you want to here
    p = plot_auc_by_time_from_first_visit(
        first_visit_timestamps=first_visit_timestamp,
    )

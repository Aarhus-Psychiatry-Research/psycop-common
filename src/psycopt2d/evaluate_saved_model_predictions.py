"""Helpers and example code for evaluating an already trained model based on a
pickle file with model predictions and hydra config.

Possible extensions (JIT when needed):
- Load most recent directory from 'evaluation_results'/Overtaci equivalent folder
- Evaluate all models in 'evaluation_results' folder
- CLI for evaluating a model
"""

from psycopt2d.utils import (
    PROJECT_ROOT,
    infer_outcome_col_name,
    infer_y_hat_prob_col_name,
    load_evaluation_data,
)
from psycopt2d.visualization import plot_auc_by_time_from_first_visit

if __name__ == "__main__":

    eval_path = PROJECT_ROOT / "evaluation_results"
    # change to whatever modele you wish to evaluate
    eval_data = load_evaluation_data(eval_path / "2022_10_18_13_23_2h3cxref")

    y_col_name = infer_outcome_col_name(eval_data.df)

    y_hat_prob_name = infer_y_hat_prob_col_name(eval_data.df)

    first_visit_timestamp = eval_data.df.groupby(eval_data.cfg.data.id_col_name)[
        eval_data.cfg.data.pred_timestamp_col_name
    ].transform("min")

    # Do whatever extra evaluation you want to here
    p = plot_auc_by_time_from_first_visit(
        labels=eval_data.df[y_col_name],
        y_hat_probs=eval_data.df[y_hat_prob_name],
        first_visit_timestamps=first_visit_timestamp,
        prediction_timestamps=eval_data.df[eval_data.cfg.data.pred_timestamp_col_name],
    )

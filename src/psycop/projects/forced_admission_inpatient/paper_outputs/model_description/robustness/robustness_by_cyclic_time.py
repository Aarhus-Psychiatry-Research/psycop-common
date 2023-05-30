from psycop.common.model_evaluation.binary.time.periodic_data import (
    roc_auc_by_periodic_time_df,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN
from psycop.projects.t2d.utils.best_runs import ModelRun


def auroc_by_hour_of_day(run: ModelRun):
    eval_ds = run.get_eval_dataset()

    roc_auc_by_periodic_time_df(
        labels=eval_ds.y,
        y_hat=eval_ds.y_hat_probs,
        timestamps=eval_ds.pred_timestamps,
        bin_period="H",
    )

    # TODO: Plotting function


def auroc_by_day_of_week(run: ModelRun):
    eval_ds = run.get_eval_dataset()

    roc_auc_by_periodic_time_df(
        labels=eval_ds.y,
        y_hat=eval_ds.y_hat_probs,
        timestamps=eval_ds.pred_timestamps,
        bin_period="H",
    )

    # TODO: Plotting function


def auroc_by_month_of_year(run: ModelRun):
    eval_ds = run.get_eval_dataset()

    roc_auc_by_periodic_time_df(
        labels=eval_ds.y,
        y_hat=eval_ds.y_hat_probs,
        timestamps=eval_ds.pred_timestamps,
        bin_period="H",
    )

    # TODO: Plotting function


def roc_auc_by_cyclic_time():
    print("Plotting AUC by cyclic time")
    EVAL_RUN.get_eval_dataset()


if __name__ == "__main__":
    roc_auc_by_cyclic_time()

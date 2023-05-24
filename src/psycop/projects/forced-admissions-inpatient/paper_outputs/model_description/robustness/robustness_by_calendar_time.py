from psycop.common.model_evaluation.binary.time.absolute_data import (
    create_roc_auc_by_absolute_time_df,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN


def roc_auc_by_calendar_time():
    print("Plotting AUC by calendar time")
    eval_ds = EVAL_RUN.get_eval_dataset()

    create_roc_auc_by_absolute_time_df(
        labels=eval_ds.y,
        y_hat_probs=eval_ds.y_hat_probs,
        timestamps=eval_ds.pred_timestamps,
        bin_period="Q",
        confidence_interval=True,
    )

    # TODO: Create plotting function


if __name__ == "__main__":
    roc_auc_by_calendar_time()

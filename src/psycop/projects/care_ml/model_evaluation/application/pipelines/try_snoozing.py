import datetime as dt

import pandas as pd
from care_ml.model_evaluation.best_runs import best_run
from care_ml.model_evaluation.data.load_true_data import load_eval_dataset
from care_ml.model_evaluation.snoozing import snooze_dataframe
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    evaluation_dataset = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
    )

    eval_df = pd.DataFrame(
        {
            "id": evaluation_dataset.ids,
            "pred_timestamps": evaluation_dataset.pred_timestamps,
            "y": evaluation_dataset.y,
            "y_hat_probs": evaluation_dataset.y_hat_probs,
        },
    )

    pred_threshold = evaluation_dataset.y_hat_probs.quantile(0.95)

    eval_df["y_hat_int"] = eval_df["y_hat_probs"].apply(
        lambda x: 1 if x > pred_threshold else 0,
    )

    for snooze_days in range(360, -90, -90):
        filtered_pred_times = snooze_dataframe(
            df=eval_df,
            snoozing_timedelta=dt.timedelta(days=snooze_days),
            prediction_column_name="y_hat_int",
            time_column_name="pred_timestamps",
            id_column_name="id",
        )

        filtered_eval_df = eval_df.merge(
            filtered_pred_times,
            on=["id", "pred_timestamps"],
            how="inner",
        )

        roc_auc = roc_auc_score(
            y_true=filtered_eval_df["y"],
            y_score=filtered_eval_df["y_hat_probs"],
        )

        print(f"Snooze days: {snooze_days}, ROC AUC: {roc_auc:.4f}")

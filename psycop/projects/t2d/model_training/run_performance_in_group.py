import datetime

import pandas as pd

from psycop.projects.t2d.paper_outputs.config import DEVELOPMENT_GROUP
from psycop.projects.t2d.utils.pipeline_objects import RunGroup


def get_best_models_by_lookahead(current_group: RunGroup) -> pd.DataFrame:
    all_models = current_group.all_runs_performance_df

    best_models_by_architecture_lookahead = (
        all_models.sort_values("roc_auc", ascending=False)
        .groupby(["model_name", "lookahead_days"])
        .head(1)
        .sort_values(["model_name", "lookahead_days"])
    )[["model_name", "lookahead_days", "roc_auc", "run_name", "timestamp"]]

    return best_models_by_architecture_lookahead


if __name__ == "__main__":
    print("\n\n")
    run_performance_df = get_best_models_by_lookahead(current_group=DEVELOPMENT_GROUP)

    now = datetime.datetime.now()

    try:
        best_in_last_hour = (
            run_performance_df[run_performance_df["timestamp"] > now - datetime.timedelta(hours=1)]
            .sort_values("roc_auc", ascending=False)
            .head(1)
            .reset_index(drop=True)["roc_auc"][0]
        )

        best_before_last_hour = (
            run_performance_df[run_performance_df["timestamp"] < now - datetime.timedelta(hours=1)]
            .sort_values("roc_auc", ascending=False)
            .head(1)
            .reset_index(drop=True)["roc_auc"][0]
        )
    except KeyError:
        best_in_last_hour = None
        best_before_last_hour = None

    if best_before_last_hour is not None and best_in_last_hour is not None:
        improvement_over_last_hour = best_in_last_hour - best_before_last_hour
        auroc_improvement_threshold = 0.001
        if improvement_over_last_hour < auroc_improvement_threshold:
            print(
                f"---- READY TO TERMINATE: Improvement of {improvement_over_last_hour} is smaller than threshold of {auroc_improvement_threshold} ----"
            )
        else:
            print(f"AUROC improvement over last hour was {improvement_over_last_hour}")

    first_model_timestamp = run_performance_df.sort_values("timestamp", ascending=True).head(1)[
        "timestamp"
    ][0]

    training_minutes = round((now - first_model_timestamp).total_seconds() / 60)  # type: ignore
    print(f"Model training has been going on for {training_minutes} minutes")

    models_trained_total = len(run_performance_df)
    print(f"In total, {models_trained_total} models have been trained")
    print("\n")

    models_trained_by_architecure_and_lookahead = run_performance_df.groupby(
        ["model_name", "lookahead_days"]
    ).count()["run_name"]

    print(run_performance_df)

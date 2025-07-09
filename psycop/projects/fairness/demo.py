from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import precision_score, roc_auc_score

from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_disk,
)

if __name__ == "__main__":
    eval_ds = scz_bp_get_eval_ds_from_disk(
        experiment_path="E:/shared_resources/scz_bp/testing", model_type="joint"
    )
    y_hat = eval_ds.get_predictions_for_positive_rate(desired_positive_rate=0.4)[0]

    metrics = {
        "AUROC": roc_auc_score,
        "Positive predictive value": precision_score,
        "False positive rate": false_positive_rate,
        "False negative rate": false_negative_rate,
        "Selection rate": selection_rate,
        "Count": count,
    }
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=eval_ds.y,
        y_pred=y_hat,
        sensitive_features=eval_ds.is_female.replace({True: "Female", False: "Male"}),
    )

    bar_plot = metric_frame.by_group.plot.bar(
        subplots=True,
        layout=[3, 2],
        legend=False,
        figsize=[12, 8],
        title="Prediction of schizophrenia and bipolar disorder",
        colormap="Dark2",
        xlabel="Sex",
    )

    bar_plot[0][0].figure.savefig("bar_plot.png")

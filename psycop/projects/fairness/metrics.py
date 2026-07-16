from typing import Any, Literal

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_negative_rate,
    true_positive_rate,
)
from sklearn.metrics import precision_score, roc_auc_score

from psycop.common.cross_experiments.cross_project_catalogue import ModelCatalogue
from psycop.common.model_evaluation.binary.utils import auroc_by_group
from psycop.projects.fairness.bootstrap import cluster_bootstrap
from psycop.projects.fairness.getters import get_eval_dfs


def by_patient(eval_df: pd.DataFrame) -> pd.DataFrame:
    # eval_df.groupby(["model", "dw_ek_borger", "sex"])[["y_hat_prob", "age"]].mean().reset_index()
    # eval_df.groupby(["model", "dw_ek_borger", "sex"])[["y", "y_hat"]].max().reset_index()

    eval_df["hits"] = np.where((eval_df["y"] == 1) & (eval_df["y_hat"] == 1), 1, 0)
    patient_df = (
        eval_df.groupby(["model", "dw_ek_borger", "sex"], as_index=False)
        .agg({"y_hat_prob": "mean", "age_group": "first", "y": "max", "hits": "max"})
        .rename(columns={"hits": "y_hat"})
    )

    # patient_df["age_group"] = bin_continuous_data(
    #         series=patient_df["age"],
    #         bins=[0, 17, 25, 40, 55, 70],
    #     )[0]
    # patient_df["age_group"] = patient_df["age_group"].replace("0-17", "<18")

    return patient_df


def add_group_prevalence(eval_df: pd.DataFrame, protected_attribute: str) -> pd.DataFrame:
    prevalences = (
        eval_df.groupby(["model", protected_attribute], as_index=False)["y"]
        .mean()
        .rename(columns={"y": "prevalence"})
    )  # type: ignore

    return prevalences


def get_metrics(
    eval_df: pd.DataFrame,
    metrics: dict[str, Any],
    protected_attribute: Literal["sex", "age_group", "region", "unit"],
) -> pd.DataFrame:
    eval_df = eval_df[eval_df[protected_attribute].notna()]
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=eval_df["y"],
        y_pred=eval_df["y_hat"],
        sensitive_features=eval_df[protected_attribute],
        control_features=eval_df["model"],
        n_boot=100,
        ci_quantiles=[0.025, 0.975],
    )

    mean = metric_frame.by_group.reset_index()
    mean = mean.melt(
        id_vars=["model", protected_attribute, "Count"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
    )
    lower = metric_frame.by_group_ci[0].reset_index()
    lower = lower.melt(
        id_vars=["model", protected_attribute],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
        value_name="lower",
    )
    upper = metric_frame.by_group_ci[1].reset_index()
    upper = upper.melt(
        id_vars=["model", protected_attribute],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
        value_name="upper",
    )

    metric_df = mean.merge(lower, how="left", on=["model", protected_attribute, "variable"])
    metric_df = metric_df.merge(upper, how="left", on=["model", protected_attribute, "variable"])

    auroc_df = auroc_by_group(
        eval_df.rename(columns={"y_hat_prob": "y_hat_probs"}),
        ["model", protected_attribute],
        stratified=True,
    )  # type: ignore

    auroc_df = auroc_df.rename(
        columns={"auroc": "value", "ci_lower": "lower", "ci_upper": "upper", "n_in_bin": "Count"}
    ).drop(columns=["level_2"])
    auroc_df["variable"] = "AUROC"

    return pd.concat([metric_df, auroc_df])


def get_bootstraps(
    eval_df: pd.DataFrame,
    protected_attribute: Literal["sex", "age_group", "region", "unit"],
    n_bootstrap: int = 100,
    rng: int | None = None,
) -> pd.DataFrame:
    eval_df = eval_df[eval_df[protected_attribute].notna()]

    dfs = []
    for _ in range(n_bootstrap):
        dfs.append(
            cluster_bootstrap(
                eval_df,
                rng=np.random.default_rng(rng if rng is not None else np.random.default_rng()),
                sampling_unit_col="dw_ek_borger",
                stratify_col=protected_attribute,
                sample_weight=True,
            )
        )

    return pd.concat(dfs)


def _get_metrics(
    eval_df: pd.DataFrame,
    metrics: dict[str, Any],
    protected_attribute: Literal["sex", "age_group", "region", "unit"],
) -> pd.DataFrame:
    eval_df = eval_df[eval_df[protected_attribute].notna()]
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=eval_df["y"],
        y_pred=eval_df["y_hat"],
        sensitive_features=eval_df[protected_attribute],
    )

    mean = metric_frame.by_group.reset_index()
    mean = mean.melt(
        id_vars=[protected_attribute, "Count"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
    )
    lower = metric_frame.by_group_ci[0].reset_index()
    lower = lower.melt(
        id_vars=[protected_attribute],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
        value_name="lower",
    )
    upper = metric_frame.by_group_ci[1].reset_index()
    upper = upper.melt(
        id_vars=[protected_attribute],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
        value_name="upper",
    )

    metric_df = mean.merge(lower, how="left", on=[protected_attribute, "variable"])
    metric_df = metric_df.merge(upper, how="left", on=[protected_attribute, "variable"])

    auroc_frame = MetricFrame(
        metrics={"AUROC": roc_auc_score},
        y_true=eval_df["y"],
        y_pred=eval_df["y_pred"],
        sensitive_features=eval_df[protected_attribute],
    )

    auroc_df = auroc_df.rename(
        columns={"auroc": "value", "ci_lower": "lower", "ci_upper": "upper", "n_in_bin": "Count"}
    ).drop(columns=["level_2"])
    auroc_df["variable"] = "AUROC"

    return pd.concat([metric_df, auroc_df])


if __name__ == "__main__":
    eval_df = get_eval_dfs(
        ModelCatalogue(projects=["CVD", "ECT", "FAI", "Restraint", "SCZ_BP", "T2D"])
    )

    metrics = {
        "Positive predictive value": precision_score,
        "True positive rate": true_positive_rate,
        "True negative rate": true_negative_rate,
        "False positive rate": false_positive_rate,
        "False negative rate": false_negative_rate,
        "Selection rate": selection_rate,
        "Count": count,
    }

    cvd = get_eval_dfs(ModelCatalogue(projects=["CVD"]))

    cvd_boots = get_bootstraps(cvd, protected_attribute="sex", n_bootstrap=100, rng=42)

    cvd_metrics = get_metrics(cvd, metrics, protected_attribute="sex")

    protected_attribute = "unit"
    metric_df = get_metrics(eval_df, metrics, protected_attribute=protected_attribute)
    prevalences = add_group_prevalence(eval_df, protected_attribute=protected_attribute)

    metric_df = metric_df.merge(prevalences, on=["model", protected_attribute])

    metric_df["cohort_n"] = metric_df.groupby(["model", "variable"])["Count"].transform("sum")
    metric_df["proportion"] = metric_df["Count"] / metric_df["cohort_n"]

    metric_df.to_csv(f"df_{protected_attribute}.csv")

    patient_df = by_patient(eval_df)
    p_metric_df = get_metrics(patient_df, metrics, protected_attribute=protected_attribute)
    p_prevalences = add_group_prevalence(patient_df, protected_attribute=protected_attribute)

    p_metric_df = p_metric_df.merge(p_prevalences, on=["model", protected_attribute])

    p_metric_df["cohort_n"] = p_metric_df.groupby(["model", "variable"])["Count"].transform("sum")
    p_metric_df["proportion"] = p_metric_df["Count"] / p_metric_df["cohort_n"]

    p_metric_df.to_csv(f"p_df_{protected_attribute}.csv")

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
    true_positive_rate
)
from sklearn.metrics import precision_score, roc_auc_score

from psycop.common.cross_experiments.cross_project_catalogue import ModelCatalogue
from psycop.common.model_evaluation.binary.utils import auroc_by_group
from psycop.projects.fairness.bootstrap import cluster_bootstrap
from psycop.projects.fairness.getters import get_eval_dfs
from psycop.projects.fairness.utils import na_auroc, na_negative_metric, na_positive_metric, na_precision


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


def bootstrap_metrics(
    eval_df: pd.DataFrame,
    protected_attribute: Literal["sex", "age_group", "region", "unit"],
    metrics: dict[str, Any],
    n_bootstrap: int = 100,
    rng: np.random.Generator | None = None,
    sample_weight: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_group = []
    ratios = []
    overall = []
    for _ in range(n_bootstrap):
        df = cluster_bootstrap(
                eval_df,
                rng=(rng if rng is not None else np.random.default_rng()),
                sampling_unit_col="dw_ek_borger",
                stratify_col=protected_attribute,
                sample_weight=sample_weight,
            )
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=df["y"],
            y_pred=df["y_hat"],
            sensitive_features=df[protected_attribute],
            sample_params=(
                {
                    metric_name: {
                        "sample_weight": df["sample_weight"]
                    }
                    for metric_name in metrics if metric_name != "Count"
                }
                if sample_weight
                else None
            )
        )
        auroc_frame = MetricFrame(
            metrics={"AUROC": na_auroc},
            y_true=df["y"],
            y_pred=df["y_hat_prob"],
            sensitive_features=df[protected_attribute],
            sample_params=({
                    "AUROC": {
                        "sample_weight": df["sample_weight"]
                    }
                } if sample_weight
                else None)
        )
        by_group.append(pd.merge(metric_frame.by_group.reset_index(), auroc_frame.by_group.reset_index(), on=protected_attribute).assign(bootstrap=_))
        ratios.append(pd.concat([metric_frame.ratio().reset_index(), auroc_frame.ratio().reset_index()]).assign(bootstrap=_))
        overall.append(pd.concat([metric_frame.overall.reset_index(), auroc_frame.overall.reset_index()]).assign(bootstrap=_))

    results_df = pd.concat(by_group)
    results_df = results_df.melt(
        id_vars=[protected_attribute, "Count"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
            "AUROC"
        ],
    )

    missing = results_df.groupby(
        ["variable", protected_attribute]
        )["value"].apply(lambda x: x.isna().mean()).reset_index().rename(columns={"value": "missing"})

    results = results_df.groupby(["variable", protected_attribute])["value"].quantile([0.025, 0.975]).unstack().rename(columns={0.025: "lower", 0.975: "upper"}).reset_index()

    if sample_weight:
        eval_df["sample_weight"] = 1 / eval_df.groupby("dw_ek_borger")[
            "timestamp"
        ].transform("count")

    estimate_metrics = MetricFrame(metrics=metrics,
        y_true=eval_df["y"],
        y_pred=eval_df["y_hat"],
        sensitive_features=eval_df[protected_attribute],
        sample_params=(
            {
                metric_name: {
                    "sample_weight": eval_df["sample_weight"]
                }
                for metric_name in metrics if metric_name != "Count"
            }
            if sample_weight
            else None
        ))
    estimate_auroc = MetricFrame(metrics={"AUROC": na_auroc},
        y_true=eval_df["y"],
        y_pred=eval_df["y_hat_prob"],
        sensitive_features=eval_df[protected_attribute],
        sample_params=({
                    "AUROC": {
                        "sample_weight": eval_df["sample_weight"]
                    }
                }if sample_weight
                else None))
    
    estimates_df = pd.merge(estimate_metrics.by_group.reset_index(), estimate_auroc.by_group.reset_index(), on=protected_attribute)
    
    estimates_df = estimates_df.melt(
        id_vars=[protected_attribute, "Count"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
            "AUROC"
        ],
    )

    estimates = estimates_df.groupby(["variable", protected_attribute, "Count"])["value"].mean().reset_index()
    
    estimates = pd.merge(results, estimates, on=[protected_attribute, "variable"])

    prevalences = (
        eval_df.groupby(protected_attribute, as_index=False)["y"]
        .mean()
        .rename(columns={"y": "prevalence"})
    ) # type: ignore
    
    estimates = pd.merge(estimates, prevalences, on=protected_attribute)

    estimates = pd.merge(estimates, missing, on=[protected_attribute, "variable"])

    if len(eval_df[protected_attribute].unique()) < 3:
        ratio_results = pd.concat(ratios).rename(columns={"index": "variable", 0: "value"})

        ratio_missing = ratio_results.groupby(
            ["variable"]
            )["value"].apply(lambda x: x.isna().mean()).reset_index().rename(columns={"value": "missing"})

        ratio_results = ratio_results.groupby(["variable"])["value"].quantile([0.025, 0.975]).unstack().rename(columns={0.025: "lower", 0.975: "upper"}).reset_index()

        ratio_estimates = pd.concat([estimate_metrics.ratio().reset_index(), estimate_auroc.ratio().reset_index()]).rename(columns={"index": "variable", 0: "value"})

        ratio_estimates = ratio_estimates.groupby(["variable"])["value"].mean().reset_index()
        
        ratio_estimates = pd.merge(ratio_results, ratio_estimates, on="variable")

        ratio_estimates = pd.merge(ratio_estimates, ratio_missing, on="variable")

    else:
        overall_df = pd.concat(overall).rename(columns={"index": "variable", 0: "overall"})

        ratio_results = pd.merge(results_df, overall_df, how="left", on=["variable"])
        ratio_results["ratio"] = np.log(ratio_results["value"]) - np.log(ratio_results["overall"])

        ratio_missing = ratio_results.groupby([protected_attribute, "variable"])["ratio"].apply(lambda x: x.isna().mean()).reset_index().rename(columns={"ratio": "missing"})

        ratio_results= ratio_results.groupby([protected_attribute, "variable"])["ratio"].quantile([0.025, 0.975]).unstack().rename(columns={0.025: "lower", 0.975: "upper"}).reset_index()

        estimates_overall = pd.concat([estimate_metrics.overall.reset_index(), estimate_auroc.overall.reset_index()]).rename(columns={"index": "variable", 0: "overall"})
        ratio_estimates = pd.merge(estimates_df, estimates_overall, how="left", on=["variable"])
        ratio_estimates["ratio"] = ratio_estimates["value"] / ratio_estimates["overall"]

        ratio_estimates = ratio_estimates.drop(columns=["value", "overall"]).rename(columns={"ratio": "value"})
                
        ratio_estimates = pd.merge(ratio_results, ratio_estimates, on=[protected_attribute, "variable"])
        
        ratio_estimates = pd.merge(ratio_estimates, ratio_missing, on=[protected_attribute, "variable"])


    return estimates, ratio_estimates




if __name__ == "__main__":
    eval_df = get_eval_dfs(
        ModelCatalogue(projects=["CVD", "ECT", "FAI", "Restraint", "SCZ_BP", "T2D"])
    )

    metrics = {
        "Positive predictive value": na_precision,
        "True positive rate": na_positive_metric(true_positive_rate),
        "True negative rate": na_negative_metric(true_negative_rate),
        "False positive rate": na_negative_metric(false_positive_rate),
        "False negative rate": na_positive_metric(false_negative_rate),
        "Selection rate": selection_rate,
        "Count": count,
    }


    protected_attribute = "age_group"

    by_groups = []
    ratios = []
    for model in eval_df.model.unique():
        by_group, ratio = bootstrap_metrics(eval_df[eval_df["model"] == model], metrics=metrics, protected_attribute=protected_attribute, n_bootstrap=1000, rng=np.random.default_rng(42), sample_weight=False)
        by_groups.append(by_group.assign(model=model))
        ratios.append(ratio.assign(model=model))

    by_group_boot = pd.concat(by_groups)
    ratio_boot = pd.concat(ratios)

    by_group_boot["cohort_n"] = by_group_boot.groupby(["model", "variable"])["Count"].transform("sum")
    by_group_boot["proportion"] = by_group_boot["Count"] / by_group_boot["cohort_n"]

    by_group_boot.to_csv(f"metrics_{protected_attribute}_1000.csv")
    ratio_boot.to_csv(f"ratios_{protected_attribute}_1000.csv")

    
    cvd_boots = bootstrap_metrics(eval_df[eval_df["model"] == "CVD"], metrics=metrics, protected_attribute="sex", n_bootstrap=100, rng=np.random.default_rng(42), sample_weight=False).assign(model="CVD")
    ect_boots = bootstrap_metrics(eval_df[eval_df["model"] == "ECT"], metrics=metrics, protected_attribute="sex", n_bootstrap=100, rng=np.random.default_rng(42), sample_weight=True).assign(model="ECT")
    t2d_boots = bootstrap_metrics(eval_df[eval_df["model"] == "T2D"], metrics=metrics, protected_attribute="sex", n_bootstrap=100, rng=np.random.default_rng(42), sample_weight=True).assign(model="T2D")
    sczbp_boots = bootstrap_metrics(eval_df[eval_df["model"] == "SCZ/BP"], metrics=metrics, protected_attribute="sex", n_bootstrap=100, rng=np.random.default_rng(42), sample_weight=True).assign(model="SCZ/BP")
    pr_boots = bootstrap_metrics(eval_df[eval_df["model"] == "PR"], metrics=metrics, protected_attribute="sex", n_bootstrap=100, rng=np.random.default_rng(42), sample_weight=True).assign(model="PR")
    ivc_boots = bootstrap_metrics(eval_df[eval_df["model"] == "IVC"], metrics=metrics, protected_attribute="sex", n_bootstrap=100, rng=np.random.default_rng(42), sample_weight=True).assign(model="IVC")

    metric_df = pd.concat([cvd_boots, ect_boots, t2d_boots, sczbp_boots, pr_boots, ivc_boots])

    metric_df["cohort_n"] = metric_df.groupby(["model", "variable"])["Count"].transform("sum")
    metric_df["proportion"] = metric_df["Count"] / metric_df["cohort_n"]

    metric_df.to_csv("metrics_sex_weighted.csv")




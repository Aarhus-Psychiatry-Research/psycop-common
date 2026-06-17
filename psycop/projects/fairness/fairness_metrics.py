from typing import Any, Literal

import pandas as pd
import numpy as np

from psycop.common.cross_experiments.cross_project_catalogue import ModelCatalogue
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_negative_rate,
    true_positive_rate,
)
from sklearn.metrics import precision_score

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.model_evaluation.binary.utils import auroc_by_group
from psycop.common.model_evaluation.utils import bin_continuous_data
from psycop.common.model_training.training_output.dataclasses import get_predictions_for_positive_rate

def get_eval_dfs(catalogue: ModelCatalogue) -> pd.DataFrame:
    eval_dfs = catalogue.get_eval_dfs()
    pprs = catalogue.get_predicted_positive_rates()

    cvd = eval_dfs["CVD"]
    cvd["y_hat"] = get_predictions_for_positive_rate(pprs["CVD"], cvd["y_hat_prob"])[0]
    cvd["dw_ek_borger"] = cvd["pred_time_uuid"].str.split("-").str[0].astype("int64")
    cvd["timestamp"] = cvd["pred_time_uuid"].str.split("-").apply(lambda x: "-".join(x[1:])).pipe(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
    cvd["model"] = "Cardiovascular disease"

    ect = eval_dfs["ECT"]
    ect["y_hat"] = get_predictions_for_positive_rate(pprs["ECT"], ect["y_hat_prob"])[0]
    ect["dw_ek_borger"] = ect["pred_time_uuid"].str.split("-").str[0].astype("int64")
    ect["timestamp"] = ect["pred_time_uuid"].str.split("-").apply(lambda x: "-".join(x[1:])).pipe(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
    ect["model"] = "Electroconvulsive therapy"

    restraint = eval_dfs["Restraint"]
    restraint["y_hat"] = get_predictions_for_positive_rate(pprs["Restraint"], restraint["y_hat_prob"])[0]
    restraint["dw_ek_borger"] = restraint["pred_time_uuid"].str.split("-").str[0].astype("int64")
    restraint["timestamp"] = restraint["pred_time_uuid"].str.split("-").apply(lambda x: "-".join(x[1:])).pipe(pd.to_datetime, format="%Y-%m-%d-%H-%M-%S")
    restraint["model"] = "Physical restraint"

    sczbp = eval_dfs["SCZ_BP"]
    sczbp["y_hat"] = get_predictions_for_positive_rate(pprs["SCZ_BP"], sczbp["y_hat_prob"])[0]
    sczbp["dw_ek_borger"] = sczbp["pred_time_uuid"].str.split("-").str[0].astype("int64")
    sczbp["timestamp"] = sczbp["pred_time_uuid"].str.split("-").apply(lambda x: "-".join(x[1:])).pipe(pd.to_datetime, format="%Y-%m-%d-%H-%M-%S")
    sczbp["model"] = "Scizophrenia or bipolar disorder"

    fai = eval_dfs["FAI"]
    fai["y_hat"] = get_predictions_for_positive_rate(pprs["FAI"], fai["y_hat_prob"])[0]
    fai["dw_ek_borger"] = fai["pred_time_uuid"].str.split("-").str[0].astype("int64")
    fai["timestamp"] = fai["pred_time_uuid"].str.split("-").apply(lambda x: "-".join(x[1:])).pipe(pd.to_datetime, format="%Y-%m-%d-%H-%M-%S")
    fai["model"] = "Involuntary hospitalisation"

    t2d = pd.DataFrame(
        {
            "y": eval_dfs["T2D"]["y"],
            "y_hat_prob": eval_dfs["T2D"]["y_hat_probs"],
            "pred_time_uuid": eval_dfs["T2D"]["pred_time_uuids"],
            "y_hat": get_predictions_for_positive_rate(pprs["T2D"], eval_dfs["T2D"]["y_hat_probs"])[0],
            "dw_ek_borger": eval_dfs["T2D"]["ids"],
            "timestamp": eval_dfs["T2D"]["pred_timestamps"],
            "model": "Type 2 diabetes"
        }
    )

    eval_df = pd.concat([cvd, ect, restraint, sczbp, fai, t2d])
    
    eval_df = eval_df.merge(birthdays(), on="dw_ek_borger", how="left")
    eval_df["age"] = (eval_df["timestamp"] - eval_df["date_of_birth"]).dt.total_seconds() / (60 * 60 * 24)
    eval_df["age"] = (eval_df["age"] / 365.25).astype("int")
    eval_df["age_group"] = bin_continuous_data(
            series=eval_df["age"],
            bins=[18, 25, 40, 55, 70],
        )[0]
    eval_df = eval_df[eval_df["age_group"].notna()]

    eval_df = eval_df.merge(sex_female(), on="dw_ek_borger", how="left")
    eval_df["sex"] = eval_df["sex_female"].replace({True: "Female", False: "Male"})

    return eval_df.drop(columns=["pred_time_uuid", "date_of_birth", "sex_female"])

def by_patient(eval_df: pd.DataFrame) -> pd.DataFrame:
    # eval_df.groupby(["model", "dw_ek_borger", "sex"])[["y_hat_prob", "age"]].mean().reset_index()
    # eval_df.groupby(["model", "dw_ek_borger", "sex"])[["y", "y_hat"]].max().reset_index()
    

    eval_df["hits"] = np.where((eval_df["y"] == 1) & (eval_df["y_hat"] == 1), 1, 0)
    patient_df = eval_df.groupby(["model", "dw_ek_borger", "sex"], as_index=False).agg({
        "y_hat_prob": "mean",
        "age_group": "first",
        "y": "max",
        "hits": "max"}).rename(columns={"hits": "y_hat"})
    
    # patient_df["age_group"] = bin_continuous_data(
    #         series=patient_df["age"],
    #         bins=[0, 17, 25, 40, 55, 70],
    #     )[0]
    # patient_df["age_group"] = patient_df["age_group"].replace("0-17", "<18")

    
    return patient_df 

def add_group_prevalence(eval_df: pd.DataFrame, protected_attribute: str) -> pd.DataFrame:
    prevalences = eval_df.groupby(["model", protected_attribute], as_index=False)["y"].mean().rename(columns={"y": "prevalence"}) # type: ignore

    return prevalences

def get_metrics(eval_df: pd.DataFrame, metrics: dict[str, Any], protected_attribute: Literal["sex", "age_group"]) -> pd.DataFrame:
    metric_frame = MetricFrame(metrics=metrics,
        y_true=eval_df["y"],
        y_pred=eval_df["y_hat"],
        sensitive_features=eval_df[protected_attribute],
        control_features=eval_df["model"],
        n_boot=100,
        ci_quantiles=[0.025, 0.975])

    mean=metric_frame.by_group.reset_index()
    mean=mean.melt(
        id_vars=["model", protected_attribute, "Count"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate"
        ],
    )
    lower=metric_frame.by_group_ci[0].reset_index()
    lower=lower.melt(
        id_vars=["model", protected_attribute],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate"
        ],
        value_name="lower"
    )
    upper=metric_frame.by_group_ci[1].reset_index()
    upper=upper.melt(
        id_vars=["model", protected_attribute],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate"
        ],
        value_name="upper"
    )

    metric_df = mean.merge(lower, how="left", on=["model", protected_attribute, "variable"])
    metric_df = metric_df.merge(upper, how="left", on=["model", protected_attribute, "variable"])

    auroc_df = auroc_by_group(eval_df.rename(columns={"y_hat_prob": "y_hat_probs"}), ["model", protected_attribute], stratified=True) # type: ignore
    
    auroc_df = auroc_df.rename(columns={"auroc": "value", "ci_lower": "lower", "ci_upper": "upper", "n_in_bin": "Count"}).drop(columns=["level_2"])
    auroc_df["variable"] = "AUROC"

    return pd.concat([metric_df, auroc_df])

if __name__ == "__main__":
    eval_df = get_eval_dfs(ModelCatalogue(projects=["CVD", "ECT", "FAI", "Restraint", "SCZ_BP", "T2D"]))
    
    metrics = {
        "Positive predictive value": precision_score,
        "True positive rate": true_positive_rate,
        "True negative rate": true_negative_rate,
        "False positive rate": false_positive_rate,
        "False negative rate": false_negative_rate,
        "Selection rate": selection_rate,
        "Count": count,
    }

    protected_attribute = "sex"
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


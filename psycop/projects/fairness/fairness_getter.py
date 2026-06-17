import plotnine as pn
import pandas as pd
from psycop.common.cross_experiments.project_getters.cvd_getter import CVDGetter
from psycop.common.cross_experiments.project_getters.ect_getter import ECTGetter
from psycop.common.cross_experiments.project_getters.restraint_getter import RestraintGetter
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.model_evaluation.utils import bin_continuous_data
from psycop.common.model_training.training_output.dataclasses import get_predictions_for_positive_rate
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import scz_bp_get_eval_ds_from_disk
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


def get_eval_df() -> pd.DataFrame:
    cvd_getter = CVDGetter()
    cvd_eval = cvd_getter.get_eval_df()
    cvd_eval["y_hat"] = get_predictions_for_positive_rate(cvd_getter.predicted_positive_rate, cvd_eval["y_hat_prob"])[0]
    cvd_eval["dw_ek_borger"] = cvd_eval["pred_time_uuid"].str.split("-").str[0].astype("int64")
    cvd_eval["timestamp"] = cvd_eval["pred_time_uuid"].str.split("-").apply(lambda x: "-".join(x[1:])).pipe(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
    cvd_eval["model"] = "CVD"

    ect_getter = ECTGetter()
    ect_eval = ect_getter.get_eval_df()
    ect_eval["y_hat"] = get_predictions_for_positive_rate(ect_getter.predicted_positive_rate, ect_eval["y_hat_prob"])[0]
    ect_eval["dw_ek_borger"] = ect_eval["pred_time_uuid"].str.split("-").str[0].astype("int64")
    ect_eval["timestamp"] = ect_eval["pred_time_uuid"].str.split("-").apply(lambda x: "-".join(x[1:])).pipe(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
    ect_eval["model"] = "ECT"

    restraint_getter = RestraintGetter()
    restraint_eval = restraint_getter.get_eval_df()
    restraint_eval["y_hat"] = get_predictions_for_positive_rate(restraint_getter.predicted_positive_rate, restraint_eval["y_hat_prob"])[0]
    restraint_eval["dw_ek_borger"] = restraint_eval["pred_time_uuid"].str.split("-").str[0].astype("int64")
    restraint_eval["timestamp"] = restraint_eval["pred_time_uuid"].str.split("-").apply(lambda x: "-".join(x[1:])).pipe(pd.to_datetime, format="%Y-%m-%d-%H-%M-%S")
    restraint_eval["model"] = "Restraint"

    eval_ds = scz_bp_get_eval_ds_from_disk(
        experiment_path="E:/shared_resources/scz_bp/testing", model_type="joint"
    )
    scz_bp_eval = pd.DataFrame(
        {
            "y": list(eval_ds.y),
            "y_hat_prob": list(eval_ds.y_hat_probs),
            "y_hat": eval_ds.get_predictions_for_positive_rate(desired_positive_rate=0.04)[0],
            "dw_ek_borger": list(eval_ds.ids),
            "timestamp": eval_ds.pred_timestamps
        }
    )
    scz_bp_eval["model"] = "SCZ/BP"

    eval_df = pd.concat([cvd_eval, ect_eval, restraint_eval, scz_bp_eval])

    eval_df = eval_df.drop(columns="pred_time_uuid")
    sex_df = sex_female()
    eval_df = eval_df.merge(sex_df, on="dw_ek_borger", how="left")

    birthday_df = birthdays()
    eval_df = eval_df.merge(birthday_df, on="dw_ek_borger", how="left")
    eval_df["age"] = (eval_df["timestamp"] - eval_df["date_of_birth"]).dt.total_seconds() / (60 * 60 * 24)
    eval_df["age"] = round((eval_df["age"] / 365.25), 2)
    eval_df["age"] = bin_continuous_data(
            series=eval_df["age"],
            bins=[18, 25, 40, 55, 70],
        )[0]
    return eval_df.drop(columns="date_of_birth")

def get_metrics(df: pd.DataFrame):
    metrics = {
        "Positive predictive value": precision_score,
        "True positive rate": true_positive_rate,
        "True negative rate": true_negative_rate,
        "False positive rate": false_positive_rate,
        "False negative rate": false_negative_rate,
        "Selection rate": selection_rate,
        "Count": count,
    }

    sex_frame = MetricFrame(metrics=metrics,
                y_true=df["y"],
                y_pred=df["y_hat"],
                sensitive_features={"sex": df["sex_female"]},
                control_features=df["model"],
                n_boot=100,
                ci_quantiles=[0.025, 0.975])
    
    age_frame = MetricFrame(metrics=metrics,
                y_true=df["y"],
                y_pred=df["y_hat"],
                sensitive_features={"age": df["age"].astype(str)},
                control_features=df["model"],
                n_boot=100,
                ci_quantiles=[0.025, 0.975])
    
    metric_frame = MetricFrame(metrics=metrics,
                y_true=df["y"],
                y_pred=df["y_hat"],
                sensitive_features={"sex": df["sex_female"], "age": df["age"].astype(str)},
                control_features=df["model"],
                n_boot=100,
                ci_quantiles=[0.025, 0.975])
    
    sex_mean=sex_frame.by_group.reset_index()
    sex_mean=sex_mean.melt(
        id_vars=["model", "sex"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
    )
    sex_lower=sex_frame.by_group_ci[0].reset_index()
    sex_lower=sex_lower.melt(
        id_vars=["model", "sex"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
        value_name="lower"
    )
    sex_upper=sex_frame.by_group_ci[1].reset_index()
    sex_upper=sex_upper.melt(
        id_vars=["model", "sex"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
        value_name="upper"
    )

    sex_df = sex_mean.merge(sex_lower, how="left", on=["model", "sex", "variable"])
    sex_df = sex_df.merge(sex_upper, how="left", on=["model", "sex", "variable"])

    age_mean=age_frame.by_group.reset_index()
    age_mean=age_mean.melt(
        id_vars=["model", "age"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
    )
    age_lower=age_frame.by_group_ci[0].reset_index()
    age_lower=age_lower.melt(
        id_vars=["model", "age"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
        value_name="lower"
    )
    age_upper=age_frame.by_group_ci[1].reset_index()
    age_upper=age_upper.melt(
        id_vars=["model", "age"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
        value_name="upper"
    )

    age_df = age_mean.merge(age_lower, how="left", on=["model", "age", "variable"])
    age_df = age_df.merge(age_upper, how="left", on=["model", "age", "variable"])

    metric_mean=metric_frame.by_group.reset_index()
    metric_mean=metric_mean.melt(
        id_vars=["model", "age", "sex"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
    )
    metric_lower=metric_frame.by_group_ci[0].reset_index()
    metric_lower=metric_lower.melt(
        id_vars=["model", "age", "sex"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
        value_name="lower"
    )
    metric_upper=metric_frame.by_group_ci[1].reset_index()
    metric_upper=metric_upper.melt(
        id_vars=["model", "age", "sex"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
        value_name="upper"
    )

    metric_df = metric_mean.merge(metric_lower, how="left", on=["model", "age", "sex", "variable"])
    metric_df = metric_df.merge(metric_upper, how="left", on=["model", "age", "sex", "variable"])

    age_df.to_csv("age_metrics.csv")
    metric_df.to_csv("metrics_metrics.csv")
    pass

def plotnine_metric_frame(df: pd.DataFrame):
    p = (
        pn.ggplot(df, pn.aes(x="variable", y="value", ymin="lower", ymax="upper", fill="model", alpha="sex"))
        + pn.geom_bar(
            stat="identity", position="dodge"
        )  # pn.aes(x="sex", y="proportion_of_n", fill="sex"),
        #+ pn.geom_path(group=1, size=1)
        #+ pn.labs(x="Sex", y="AUROC", title="Comparison of AUROC and other metrics")
        + pn.geom_errorbar(stat="identity", position="dodge", width=0.9)
        + pn.scale_alpha_manual(values=[1, 0.3])
        + pn.ylim(0, 1)
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            #axis_text_y=pn.element_text(size=15),
            panel_grid_minor=pn.element_blank(),
            #legend_position="none",
            axis_title=pn.element_blank(),
            #plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
            # figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        # + pn.scale_fill_manual(values=["#669BBC", "#669BBC"])
    )
    p.save("test.png")

    p = (
        pn.ggplot(age_df[age_df["variable"] == "Positive predictive value"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", fill="age"))
        + pn.geom_bar(
            stat="identity", position="dodge"
        )  # pn.aes(x="sex", y="proportion_of_n", fill="sex"),
        #+ pn.geom_path(group=1, size=1)
        + pn.labs(x="Model", y="Value", title="Positive predictive value")
        + pn.geom_errorbar(stat="identity", position="dodge", width=0.9)
        # + pn.ylim(0, 1)
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            #axis_text_y=pn.element_text(size=15),
            panel_grid_minor=pn.element_blank(),
            #legend_position="none",
            axis_title=pn.element_blank(),
            #plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
            # figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        # + pn.scale_fill_manual(values=["#669BBC", "#669BBC"])
    )
    p.save("test_age.png")

    p = (
        pn.ggplot(metric_df[metric_df["variable"] == "Positive predictive value"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", fill="age", alpha="sex"))
        + pn.geom_bar(
            stat="identity", position="dodge"
        )  # pn.aes(x="sex", y="proportion_of_n", fill="sex"),
        #+ pn.geom_path(group=1, size=1)
        + pn.labs(x="Model", y="Value", title="Positive predictive value")
        + pn.geom_errorbar(stat="identity", position="dodge", width=0.9)
        # + pn.ylim(0, 1)
        + pn.scale_alpha_manual(values=[1, 0.3])
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            #axis_text_y=pn.element_text(size=15),
            panel_grid_minor=pn.element_blank(),
            #legend_position="none",
            axis_title=pn.element_blank(),
            #plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
            # figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        # + pn.scale_fill_manual(values=["#669BBC", "#669BBC"])
    )
    p.save("test_int.png")


if __name__ == "__main__":
    df = get_eval_df()
    get_metrics(df)

    pass

    

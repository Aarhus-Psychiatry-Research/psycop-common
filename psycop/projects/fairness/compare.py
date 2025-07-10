import numpy as np
import pandas as pd
import plotnine as pn
import polars as pl
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

from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.common.model_evaluation.binary.global_performance.roc_auc import bootstrap_roc
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_disk,
)


def parse_dw_ek_borger_from_uuid(
    df: pl.DataFrame, output_col_name: str = "dw_ek_borger"
) -> pl.DataFrame:
    return df.with_columns(
        pl.col("pred_time_uuid").str.split("-").list.first().cast(pl.Int64).alias(output_col_name)
    )


def scz_bp_metrics(metrics: dict[str]) -> tuple[MetricFrame, np.float16, np.float16]:  # type: ignore
    eval_ds = scz_bp_get_eval_ds_from_disk(
        experiment_path="E:/shared_resources/scz_bp/testing", model_type="joint"
    )

    y_hat = eval_ds.get_predictions_for_positive_rate(desired_positive_rate=0.4)[0]

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=eval_ds.y,
        y_pred=y_hat,
        sensitive_features=eval_ds.is_female.replace({True: "Female", False: "Male"}),  # type: ignore
        n_boot=100,
        ci_quantiles=[0.025, 0.975],
    )

    eval_df = pd.DataFrame(
        {
            "y": list(eval_ds.y),
            "y_hat_probs": list(eval_ds.y_hat_probs["y_hat_probs"]),
            "is_female": list(eval_ds.is_female),  # type: ignore
        }
    )

    auroc_male = roc_auc_score(
        eval_df[eval_df["is_female"] is False]["y"],
        eval_df[eval_df["is_female"] is False]["y_hat_probs"],
    )
    auroc_female = roc_auc_score(
        eval_df[eval_df["is_female"] is True]["y"],
        eval_df[eval_df["is_female"] is True]["y_hat_probs"],
    )
    boot_male = bootstrap_roc(
        y=eval_df[eval_df["is_female"] is False]["y"],
        y_hat_probs=eval_df[eval_df["is_female"] is False]["y_hat_probs"],
        n_bootstraps=100,
    )
    boot = bootstrap_roc(y=eval_df["y"], y_hat_probs=eval_df["y_hat_probs"], n_bootstraps=100)

    return metric_frame, auroc_male, auroc_female  # type: ignore


def restraint_metrics(metrics: dict) -> tuple[any]:  # type: ignore
    eval_df = pl.read_parquet(
        "E:/shared_resources/restraint/eval_runs/restraint_all_tuning_v2_best_run_evaluated_on_test/eval_df.parquet"
    )
    sex_df = pl.DataFrame(sex_female())

    joint_df = (
        parse_dw_ek_borger_from_uuid(eval_df).join(sex_df, on="dw_ek_borger", how="left")
    ).to_pandas()

    y_hat = get_predictions_for_positive_rate(0.01, joint_df.y_hat_prob)[0]

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=joint_df.y,
        y_pred=y_hat,
        sensitive_features=joint_df.sex_female.replace({True: "Female", False: "Male"}),
        n_boot=100,
        ci_quantiles=[0.025, 0.975],
    )

    auroc_male = roc_auc_score(
        joint_df[joint_df["sex_female"] is False]["y"],
        joint_df[joint_df["sex_female"] is False]["y_hat_prob"],
    )
    auroc_female = roc_auc_score(
        joint_df[joint_df["sex_female"] is True]["y"],
        joint_df[joint_df["sex_female"] is True]["y_hat_prob"],
    )

    return metric_frame, auroc_male, auroc_female  # type: ignore


if __name__ == "__main__":
    metrics = {
        "Positive predictive value": precision_score,
        "True positive rate": true_positive_rate,
        "True negative rate": true_negative_rate,
        "False positive rate": false_positive_rate,
        "False negative rate": false_negative_rate,
        "Selection rate": selection_rate,
        "Count": count,
    }

    scz_bp_frame = scz_bp_metrics(metrics)
    scz_bp_df = scz_bp_frame[0].by_group
    scz_bp_df["Percentage"] = scz_bp_df["Count"] / scz_bp_df["Count"].sum()

    restraint_frame = restraint_metrics(metrics)
    restraint_df = restraint_frame[0].by_group
    restraint_df["Percentage"] = restraint_df["Count"] / restraint_df["Count"].sum()

    scz_bp_df["Model"] = "Schizophrenia/bipolar disorder"
    restraint_df["Model"] = "Composite restraint"

    df = pd.concat(  # type: ignore
        [scz_bp_df, restraint_df]  # type: ignore
    )  # pd.concat([scz_bp_df.drop(columns="Count"), restraint_df.drop(columns="Count")])
    df["Sex"] = df.index

    df = df.melt(
        id_vars=["Model", "Sex"],
        value_vars=[
            "Positive predictive value",
            "True positive rate",
            "True negative rate",
            "False positive rate",
            "False negative rate",
            "Selection rate",
        ],
    )

    scz_bp_lower = scz_bp_frame[0].by_group_ci[0]
    scz_bp_lower["Model"] = "Schizophrenia/bipolar disorder"

    scz_bp_upper = scz_bp_frame[0].by_group_ci[1]
    scz_bp_upper["Model"] = "Schizophrenia/bipolar disorder"

    restraint_lower = restraint_frame[0].by_group_ci[0]
    restraint_lower["Model"] = "Composite restraint"

    restraint_upper = restraint_frame[0].by_group_ci[1]
    restraint_upper["Model"] = "Composite restraint"

    lower = pd.concat([scz_bp_lower, restraint_lower])  # type: ignore
    lower["Sex"] = lower.index

    upper = pd.concat([scz_bp_upper, restraint_upper])  # type: ignore
    upper["Sex"] = upper.index

    df = pd.concat(  # type: ignore
        [
            df,
            pd.DataFrame(
                {
                    "Model": "Schizophrenia/bipolar disorder",
                    "Sex": "Female",
                    "variable": "AUROC",
                    "value": [scz_bp_frame[2]],
                },
                index=[24],
            ),
        ]
    )
    df = pd.concat(  # type: ignore
        [
            df,
            pd.DataFrame(
                {
                    "Model": "Schizophrenia/bipolar disorder",
                    "Sex": "Male",
                    "variable": "AUROC",
                    "value": [scz_bp_frame[1]],
                },
                index=[25],
            ),
        ]
    )

    df = pd.concat(  # type: ignore
        [
            df,
            pd.DataFrame(
                {
                    "Model": "Composite restraint",
                    "Sex": "Female",
                    "variable": "AUROC",
                    "value": [restraint_frame[2]],  # type: ignore
                },
                index=[26],
            ),
        ]
    )
    df = pd.concat(  # type: ignore
        [
            df,
            pd.DataFrame(
                {
                    "Model": "Composite restraint",
                    "Sex": "Male",
                    "variable": "AUROC",
                    "value": [restraint_frame[1]],  # type: ignore
                },
                index=[27],
            ),
        ]
    )

    lower = lower.melt(
        id_vars=["Model", "Sex"],
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

    upper = upper.melt(
        id_vars=["Model", "Sex"],
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

    df1 = df.merge(lower, how="left", on=["Model", "Sex", "variable"])

    df2 = df1.merge(upper, how="left", on=["Model", "Sex", "variable"])

    df2.to_csv("metrics_1000_auroc.csv")

    p = (
        pn.ggplot(df, pn.aes(x="variable", y="value", fill="Model", color="Sex"))
        + pn.geom_bar(
            stat="identity", position="dodge"
        )  # pn.aes(x="sex", y="proportion_of_n", fill="sex"),
        + pn.geom_path(group=1, size=1)
        + pn.labs(x="Sex", y="AUROC", title="Comparison of AUROC and other metrics")
        + pn.ylim(0, 1)
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(size=15),
            axis_text_y=pn.element_text(size=15),
            panel_grid_minor=pn.element_blank(),
            legend_position="none",
            axis_title=pn.element_text(size=22),
            plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
            figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        # + pn.scale_fill_manual(values=["#669BBC", "#669BBC"])
    )

    p.save("bar_plot.png")

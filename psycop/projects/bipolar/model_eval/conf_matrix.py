import pandas as pd
import plotnine as pn
from sklearn.metrics import roc_auc_score

from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    ConfusionMatrix,
    get_confusion_matrix_cells_from_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    EvalDataset,
    get_predictions_for_positive_rate,
)
from psycop.common.test_utils.str_to_df import str_to_df
from psycop.projects.t2d.paper_outputs.config import T2D_PN_THEME


def bp_plotnine_confusion_matrix(
    matrix: ConfusionMatrix,
    actual_outcome_text: str,
    predicted_text: str,
    auroc: float | None = None,
) -> pn.ggplot:
    """Create a confusion matrix and return a plotnine object."""
    df = str_to_df(
        f"""true,pred,estimate
+,+,"{f'{matrix.true_positives:,}'}",
+,-,"{f'{matrix.false_negatives:,}'}",
-,+,"{f'{matrix.false_positives:,}'}",
-,-,"{f'{matrix.true_negatives:,}'}",
" ",+,"PPV:\n{round(matrix.ppv*100, 1)}%",
" ",-,"NPV:\n{round(matrix.npv*100,1)}%",
-," ","Spec:\n{round(matrix.specificity*100, 1)}%",
+," ","Sens:\n{round(matrix.sensitivity*100, 1)}%",
"""
    )
    if auroc is not None:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {"true": [" "], "pred": [" "], "estimate": [f"AUROC:\n {round(auroc, 2)}"]}
                ),
            ]
        ).reset_index(drop=True)
    df["true"] = pd.Categorical(df["true"], ["-", "+", " "])
    df["pred"] = pd.Categorical(df["pred"], ["+", "-", " "])
    p = (
        pn.ggplot(df, pn.aes(x="pred", y="true", fill="estimate"))
        + T2D_PN_THEME
        + pn.geom_tile(pn.aes(width=0.95, height=0.95), fill="gainsboro")
        + pn.geom_text(pn.aes(label="estimate"), size=18, color="black")
        + pn.theme(
            axis_line=pn.element_blank(),
            axis_ticks=pn.element_blank(),
            axis_text=pn.element_text(size=15, color="black"),
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            panel_background=pn.element_blank(),
            legend_position="none",
        )
        + pn.labs(x=f"Predicted {predicted_text}", y=f"Actual {actual_outcome_text}")
    )

    return p


def bp_confusion_matrix_plot(
    y_true: pd.Series,  # type: ignore
    y_hat: pd.Series,  # type: ignore
    positive_rate: float,
    actual_outcome_text: str = "Transition from UD to BD",
    predicted_text: str = "Transition from UD to BD",
    add_auroc: bool = False,
) -> pn.ggplot:
    df = pd.DataFrame(
        {
            "true": y_true,
            "pred": get_predictions_for_positive_rate(
                y_hat_probs=y_hat, desired_positive_rate=positive_rate
            )[0],
        }
    )
    if add_auroc:
        auroc = roc_auc_score(y_true=y_true, y_score=y_hat)
    else:
        auroc = None

    confusion_matrix = get_confusion_matrix_cells_from_df(df=df)

    p = bp_plotnine_confusion_matrix(
        matrix=confusion_matrix,
        actual_outcome_text=actual_outcome_text,
        predicted_text=predicted_text,
        auroc=auroc,  # type: ignore
    )

    return p


def _output_performance_by_ppr(eval_ds: EvalDataset) -> pd.DataFrame:  # type: ignore
    df: pd.DataFrame = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=eval_ds, positive_rates=[0.07, 0.06, 0.05, 0.04, 0.03]
    )

    # df = _clean_up_performance_by_ppr(df) # noqa: ERA001
    return df


if __name__ == "__main__":
    from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper

    best_pos_rate = 0.05
    best_experiment = "bipolar_model_training_text_feature_lb_200_interval_150"
    eval_ds = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(experiment_name=best_experiment, metric="all_oof_BinaryAUROC")
        .eval_frame()
        .frame.to_pandas()
    )

    # x = _output_performance_by_ppr(eval_ds=eval_ds) # noqa: ERA001
    p = bp_confusion_matrix_plot(
        y_true=eval_ds.y,  # type: ignore
        y_hat=eval_ds.y_hat_prob,  # type: ignore
        positive_rate=best_pos_rate,  # type: ignore
    )  # type: ignore

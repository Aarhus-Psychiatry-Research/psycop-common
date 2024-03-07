import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    ConfusionMatrix,
    get_confusion_matrix_cells_from_df,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.common.test_utils.str_to_df import str_to_df
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_best_run_in_experiment,
)
from psycop.projects.t2d.paper_outputs.config import T2D_PN_THEME
from psycop.projects.t2d.paper_outputs.model_description.performance.performance_by_ppr import (
    _clean_up_performance_by_ppr,  # type: ignore
)


def scz_bp_plotnine_confusion_matrix(matrix: ConfusionMatrix, outcome_text: str) -> pn.ggplot:
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

    """Create a confusion matrix and return a plotnine object."""
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
        + pn.labs(x=f"Predicted {outcome_text}", y=f"Actual {outcome_text}")
    )

    return p


def scz_bp_confusion_matrix_plot(eval_ds: EvalDataset, positive_rate: float) -> pn.ggplot:
    df = pd.DataFrame(
        {
            "true": eval_ds.y,
            "pred": eval_ds.get_predictions_for_positive_rate(desired_positive_rate=positive_rate)[
                0
            ],
        }
    )
    confusion_matrix = get_confusion_matrix_cells_from_df(df=df)

    p = scz_bp_plotnine_confusion_matrix(
        matrix=confusion_matrix, outcome_text=f"SCZ or BP within 5 years"
    )

    return p


def _output_performance_by_ppr(eval_ds: EvalDataset) -> pd.DataFrame:  # type: ignore
    df: pd.DataFrame = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=eval_ds, positive_rates=[0.07, 0.06, 0.05, 0.04, 0.03]
    )

    # df = _clean_up_performance_by_ppr(df)
    return df


if __name__ == "__main__":
    best_experiment = "sczbp/text_only"
    best_pos_rate = 0.04
    eval_ds = scz_bp_get_eval_ds_from_best_run_in_experiment(experiment_name=best_experiment)

    # x = _output_performance_by_ppr(eval_ds=eval_ds)
    p = scz_bp_confusion_matrix_plot(eval_ds=eval_ds, positive_rate=best_pos_rate)

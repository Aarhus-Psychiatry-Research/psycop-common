from pathlib import Path

import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    ConfusionMatrix,
    get_confusion_matrix_cells_from_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.common.test_utils.str_to_df import str_to_df
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk


def plotnine_confusion_matrix(
    matrix: ConfusionMatrix, title: str = "Confusion Matrix"
) -> pn.ggplot:
    df = str_to_df(
        f"""true,pred,estimate,metric
+,+,{matrix.true_positives}," ",
+,-,{matrix.false_negatives}," ",
-,+,{matrix.false_positives}," ",
-,-,{matrix.true_negatives}," ",
" ",+," ","PPV:\n{round(matrix.ppv*100, 1)}%",
" ",-," ","NPV:\n{round(matrix.npv*100, 1)}%",
-," "," ","Specificity:\n{round(matrix.specificity*100, 1)}%",
+," "," ","Sensitivity:\n{round(matrix.sensitivity*100, 1)}%",
"""
    )

    df["true"] = pd.Categorical(df["true"], ["+", "-", " "])
    df["pred"] = pd.Categorical(df["pred"], ["+", "-", " "])
    df["fill"] = ["1", "1", "1", "1", "2", "2", "2", "2"]

    p = (
        pn.ggplot(df, pn.aes(x="true", y="pred", fill="fill"))
        + pn.geom_tile(pn.aes(width=0.95, height=0.95))
        + pn.geom_text(pn.aes(label="metric"), size=20, color="white", family="Times New Roman")
        + pn.geom_text(
            pn.aes(label="estimate"),
            size=25,
            color="white",
            fontweight="bold",
            family="Times New Roman",
        )
        + pn.theme(
            axis_line=pn.element_blank(),
            axis_ticks=pn.element_blank(),
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            panel_background=pn.element_blank(),
            legend_position="none",
            text=(pn.element_text(family="Times New Roman")),
            axis_text_x=pn.element_text(size=20, weight="bold"),
            axis_text_y=pn.element_text(size=20, weight="bold"),
            axis_title=pn.element_text(size=22),
            plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
        )
        + pn.scale_y_discrete(reverse=True)
        + pn.scale_fill_manual(values=["#B7C8B5", "#1E536D"])
        + pn.labs(title=title, y="Predicted", x="Actual")
    )

    return p


def confusion_matrix_model(df: pd.DataFrame, positive_rate: float) -> ConfusionMatrix:
    df = df.rename(columns={"y": "true", "y_hat_prob": "pred"})

    df = pd.DataFrame(
        {
            "true": df["true"],
            "pred": get_predictions_for_positive_rate(
                desired_positive_rate=positive_rate,
                y_hat_probs=df["pred"],  # type: ignore
            )[0],
        }
    )
    confusion_matrix = get_confusion_matrix_cells_from_df(df=df)

    return confusion_matrix


if __name__ == "__main__":
    save_dir = Path(__file__).parent 
    save_dir.mkdir(parents=True, exist_ok=True)

    best_experiment = "restraint_text_hyper"
    best_pos_rate = 0.05
    eval_df = read_eval_df_from_disk(
        "E:/shared_resources/restraint/eval_runs/restraint_all_tuning_best_run_evaluated_on_test"
    ).to_pandas()

    plotnine_confusion_matrix(confusion_matrix_model(df=eval_df, positive_rate=best_pos_rate)).save(
        save_dir / "restraint_confusion_matrix.png"
    )

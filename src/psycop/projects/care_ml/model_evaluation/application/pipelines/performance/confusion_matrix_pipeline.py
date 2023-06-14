from pathlib import Path

import pandas as pd
import plotnine as pn
from care_ml.model_evaluation.config import (
    COLOURS,
    EVAL_RUN,
    MODEL_NAME,
    PN_THEME,
    TABLES_PATH,
    TEXT_EVAL_RUN,
    TEXT_TABLES_PATH,
)
from care_ml.utils.best_runs import Run
from psycop.common.model_evaluation.confusion_matrix import confusion_matrix
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    ConfusionMatrix,
)
from psycop.common.test_utils.str_to_df import str_to_df


def plotnine_confusion_matrix(matrix: ConfusionMatrix, x_title: str) -> pn.ggplot:
    df = str_to_df(
        f"""true,pred,estimate
+,+,{matrix.true_positives},
+,-,{matrix.false_negatives},
-,+,{matrix.false_positives},
-,-,{matrix.true_negatives},
" ",+,"PPV:\n{matrix.ppv:.3f}",
" ",-,"NPV:\n{matrix.npv:.3f}",
-," ","Spec:\n{matrix.specificity:.3f}",
+," ","Sens:\n{matrix.sensitivity:.3f}",
""",
    )

    """Create a confusion matrix and return a plotnine object."""
    df["true"] = pd.Categorical(df["true"], ["+", "-", " "])
    df["pred"] = pd.Categorical(df["pred"], ["+", "-", " "])

    p = (
        pn.ggplot(df, pn.aes(x="true", y="pred", fill="estimate"))
        + PN_THEME
        + pn.geom_tile(pn.aes(width=0.95, height=0.95), fill=COLOURS["blue"])
        + pn.geom_text(pn.aes(label="estimate"), size=20, color="White")
        + pn.theme(
            axis_line=pn.element_blank(),
            axis_ticks=pn.element_blank(),
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            panel_background=pn.element_blank(),
            legend_position="none",
        )
        + pn.scale_y_discrete(reverse=True)
        + pn.labs(title=x_title, y="Predicted", x="Actual")
    )

    return p


def confusion_matrix_metrics(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, ConfusionMatrix]:
    """
    Creates a confusion matrix dataframe with PPV, NPV, SENS, and SPEC.
    """
    # Calculate the confusion matrix using sklearn
    cm = confusion_matrix.get_confusion_matrix_cells_from_df(df)

    # Extract the TP, FP, TN, and FN values from the confusion matrix

    # Create the confusion matrix dataframe
    df = pd.DataFrame(
        {
            "Actual Positive": [cm.true_positives, cm.false_negatives],
            "Actual Negative": [cm.false_positives, cm.true_negatives],
        },
        index=["Predicted Positive", "Predicted Negative"],
    )

    # Create a separate dataframe for the metrics
    metrics_df = pd.DataFrame(
        {
            "PPV": [cm.ppv],
            "NPV": [cm.npv],
            "SENS": [cm.sensitivity],
            "SPEC": [cm.specificity],
        },
    )

    return df, metrics_df, cm


def confusion_matrix_pipeline(run: Run, path: Path):
    eval_ds = run.get_eval_dataset()

    df = pd.DataFrame(
        {
            "true": eval_ds.y,
            "pred": eval_ds.get_predictions_for_positive_rate(
                desired_positive_rate=run.pos_rate,
            )[0],
        },
    )

    conf_matrix, metrics_df, cm = confusion_matrix_metrics(df)

    path.mkdir(parents=True, exist_ok=True)

    # Save the df to a csv file
    conf_matrix.to_csv(path / "confusion_matrix.csv")
    metrics_df.to_csv(path / "confusion_matrix_metrics.csv")

    plotnine_confusion_matrix(cm, f"Confusion Matrix for {MODEL_NAME[run.name]}").save(
        path / "confusion_matrix.png",
    )


if __name__ == "__main__":
    confusion_matrix_pipeline(EVAL_RUN, TABLES_PATH)
    confusion_matrix_pipeline(TEXT_EVAL_RUN, TEXT_TABLES_PATH)

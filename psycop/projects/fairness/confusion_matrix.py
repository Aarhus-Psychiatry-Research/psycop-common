from typing import Literal

import numpy as np

import pandas as pd
import plotnine as pn

from psycop.common.cross_experiments.cross_project_catalogue import ModelCatalogue
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import ConfusionMatrix, get_confusion_matrix_cells_from_df
from psycop.common.test_utils.str_to_df import str_to_df
from psycop.projects.fairness.getters import get_eval_dfs
from sklearn.metrics import confusion_matrix

def plotnine_confusion_matrix(
    matrix: ConfusionMatrix, title: str = "Confusion Matrix"
) -> pn.ggplot:
    df = str_to_df(
        f"""true,pred,estimate,metric
+,+,{round(matrix.true_positives, 2)}," ",
+,-,{round(matrix.false_negatives ,2)}," ",
-,+,{round(matrix.false_positives, 2)}," ",
-,-,{round(matrix.true_negatives, 2)}," ",
" ",+,"","PPV:\n{round(matrix.ppv*100, 1)}%",
" ",-,"","NPV:\n{round(matrix.npv*100, 1)}%",
-," ","","Specificity:\n{round(matrix.specificity*100, 1)}%",
+," ","","Sensitivity:\n{round(matrix.sensitivity*100, 1)}%",
"""
    )

    df["true"] = pd.Categorical(df["true"], ["+", "-", " "])
    df["pred"] = pd.Categorical(df["pred"], ["+", "-", " "])
    df["fill"] = ["1", "1", "1", "1", "2", "2", "2", "2"]
    df["estimate"] = pd.to_numeric(df["estimate"])
    
    p = (
        pn.ggplot(df, pn.aes(x="true", y="pred", fill="estimate"))
        + pn.geom_tile(pn.aes(width=0.95, height=0.95))
        + pn.geom_text(
            pn.aes(label="metric"), size=15, color="white"
        )  # , family="Times New Roman")
        + pn.geom_text(
            pn.aes(label="estimate"),
            size=25,
            color="white",
            fontweight="bold",  # family="Times New Roman",
        )
        + pn.theme(
            axis_line=pn.element_blank(),
            axis_ticks=pn.element_blank(),
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            panel_background=pn.element_blank(),
            axis_text_x=pn.element_text(size=20, weight="bold"),
            axis_text_y=pn.element_text(size=20, weight="bold"),
            axis_title=pn.element_text(size=22),
            plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
        )
        #+ pn.scale_y_discrete(reverse=True)
        #+ pn.scale_fill_manual(values=["#D3D3D3", "#808080"])
        + pn.scale_y_discrete(limits=lambda x: x[::-1])
        + pn.scale_x_discrete(limits=lambda x: x[::-1])
        + pn.scale_fill_gradient2(
        low="#6D7A5F",       # higher for females
        mid="#E2E3CC",         # equal
        high="#B8683A",      # higher for males
        midpoint=0,
        limits=(-1.5, 1.5),
        breaks=[-1.5, 0, 1.5],
        labels=[
            "Higher for females",
            "Equal",
            "Higher for males"
        ],
        name=" ")
        + pn.labs(title="CVD", y="Predicted", x="Actual")
    ).save("test.png")

    return p

def confusion_matrix_model(df: pd.DataFrame, protected_attribute: Literal["sex"]) -> ConfusionMatrix:
    levels = df[protected_attribute].unique()
    
    cb_0 = confusion_matrix(eval_df[(eval_df["model"] == "Cardiovascular disease") & (eval_df[protected_attribute] == levels[0])]["y"], eval_df[(eval_df["model"] == "Cardiovascular disease") & (eval_df[protected_attribute] == levels[0])]["y_hat"], normalize="true")

    cb_1 = confusion_matrix(eval_df[(eval_df["model"] == "Cardiovascular disease") & (eval_df[protected_attribute] == levels[1])]["y"], eval_df[(eval_df["model"] == "Cardiovascular disease") & (eval_df[protected_attribute] == levels[1])]["y_hat"], normalize="true")

    log_ratio = np.log2(cb_1 / cb_0)
    
    return ConfusionMatrix(
        true_positives=log_ratio[1][1],
        true_negatives=log_ratio[0][0],
        false_positives=log_ratio[1][0],
        false_negatives=log_ratio[0][1],
    )

if __name__ == "__main__":
    eval_df = get_eval_dfs(ModelCatalogue(projects=["CVD", "ECT", "FAI", "Restraint", "SCZ_BP", "T2D"]))

    matrix = confusion_matrix_model(eval_df[(eval_df["model"] == "Cardiovascular disease")], protected_attribute="sex")

    plotnine_confusion_matrix(matrix)
    
    pass
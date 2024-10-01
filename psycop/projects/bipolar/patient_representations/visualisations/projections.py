from typing import Literal

import pandas as pd
import plotly.express as px

from psycop.projects.bipolar.patient_representations.pca import perform_pca
from psycop.projects.bipolar.patient_representations.tsne import perform_tsne
from psycop.projects.bipolar.patient_representations.utils import prepare_eval_data_for_projections


def plot_patient_projections(
    df: pd.DataFrame,
    projecton_method: Literal["pca", "tsne"],
    cut_off_values: list[list[int | float]] | None = None,
):
    match projecton_method:
        case "pca":
            projection_df = perform_pca(df)
        case "tsne":
            projection_df = perform_tsne(df)

    if cut_off_values is not None:
        projection_df = projection_df[
            (projection_df["component_1"] > cut_off_values[0][0])
            & (projection_df["component_1"] < cut_off_values[0][1])
            & (projection_df["component_2"] > cut_off_values[1][0])
            & (projection_df["component_2"] < cut_off_values[1][1])
        ]

    # keep only patients with TN and TP
    projection_df = projection_df[projection_df["prediction_type"].isin(["TN", "TP", "FP"])]  # type: ignore

    # create a scatter plot of the PCA components with color based on prediction type
    fig = px.scatter(
        projection_df,
        x="component_1",
        y="component_2",
        color="prediction_type",
        title="Patient Projections",
        labels={"component_1": "Component 1", "component_2": "Component 2"},
    )

    d_1 = {"size": 5}

    d_2 = {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}

    # add a legend
    fig.update_traces(marker=d_1)
    fig.update_layout(legend=d_2)

    fig.show()


def plot_pca_projections(df: pd.DataFrame, cut_off_values: list[list[int | float]] | None = None):
    plot_patient_projections(df=df, projecton_method="pca", cut_off_values=cut_off_values)


def plot_tsne_projections(df: pd.DataFrame, cut_off_values: list[list[int | float]] | None = None):
    plot_patient_projections(df=df, projecton_method="tsne", cut_off_values=cut_off_values)


if __name__ == "__main__":
    df = prepare_eval_data_for_projections(
        experiment_name="bipolar_model_training_text_feature_lb_200_interval_150",
        predictor_df_name="bipolar_text_feature_set_interval_days_150",
    )
    plot_pca_projections(df, cut_off_values=[[-10000, 21000], [0.34, 0.40]])
    plot_tsne_projections(df)

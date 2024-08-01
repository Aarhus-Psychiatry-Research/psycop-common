import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline

from psycop.projects.bipolar.patient_representations.pca import perform_pca
from psycop.projects.bipolar.patient_representations.utils import (
    prepare_eval_data_for_pca,
)


def plot_patient_projections():
    df = prepare_eval_data_for_pca()
    pca_df = perform_pca(df)

    # keep only patients with TN and TP
    pca_df = pca_df[pca_df["prediction_type"].isin(["TN", "TP"])]

    # create a scatter plot of the PCA components with color based on prediction type
    fig = px.scatter(
        pca_df,
        x="component_1",
        y="component_2",
        color="prediction_type",
        title="Patient Projections",
        labels={"component_1": "Component 1", "component_2": "Component 2"},
    )
    # add a legend
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    fig.show()


if __name__ == "__main__":
    plot_patient_projections()

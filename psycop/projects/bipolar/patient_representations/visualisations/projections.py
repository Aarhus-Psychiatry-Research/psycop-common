import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from psycop.projects.bipolar.patient_representations.pca import perform_pca
from psycop.projects.bipolar.patient_representations.utils import prepare_eval_data_for_pca


def plot_patient_projections():
    df = prepare_eval_data_for_pca()
    pca_df = perform_pca(df)

    # create a scatter plot of the PCA components with color based on prediction type
    fig = px.scatter(
        pca_df,
        x="component_1",
        y="component_2",
        color="prediction_type",
        title="Patient Projections",
        labels={"component_1": "Component 1", "component_2": "Component 2"},
    )

    fig.show()


if __name__ == "__main__":
    plot_patient_projections()

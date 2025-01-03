from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px  # Import Plotly Express
import plotly.graph_objects as go
import plotly.offline

from psycop.projects.bipolar.patient_representations.representations import perform_projection
from psycop.projects.bipolar.patient_representations.utils import prepare_eval_data_for_projections


def _prepare_df_for_trajectories(
    df: pd.DataFrame,
    component_1_col_name: str = "component_1",
    component_2_col_name: str = "component_2",
    id_col_name: str = " dw_ek_borger",
    timestamp_col_name: str = "timestamp",
    label_col_name: str = "predicted_label",
    size_col_name: str = "y_pred",
    patients_to_keep: list[int] | None = None,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    int,
]:
    if patients_to_keep:
        df = df[df["dw_ek_borger"].isin(patients_to_keep)]

    # Sort DataFrame by ID
    df_sorted = df.sort_values(by=[id_col_name])

    # Transform each patients first timestamp into 1, each patients second timestamp into 2, etc.
    df_sorted["timestamp"] = df_sorted.groupby(id_col_name).cumcount() + 1

    # Make rows extra rows for patients with less than the maximum number of timestamps
    max_timestamps = df_sorted.groupby(id_col_name).timestamp.max().max()

    for patient_id in df_sorted[id_col_name].unique():
        patient_df = df_sorted[df_sorted[id_col_name] == patient_id]
        num_extra_rows = max_timestamps - patient_df.shape[0]
        if num_extra_rows > 0:
            extra_timestamps = np.arange(patient_df["timestamp"].max() + 1, max_timestamps + 1)
            # copy data from last point for all other columns
            extra_data = patient_df[patient_df.timestamp == patient_df["timestamp"].max()].copy()

            extra_data = pd.concat([extra_data] * num_extra_rows, ignore_index=True)

            extra_data["timestamp"] = extra_timestamps

            df_sorted = pd.concat([df_sorted, extra_data], ignore_index=True)

    # Sort DataFrame by timestamp and then make sure the order of the patients is preserved
    df_sorted = df_sorted.sort_values(by=[timestamp_col_name, id_col_name])

    x_data = df_sorted[component_1_col_name].values.reshape(-1, df_sorted[id_col_name].nunique())  # type: ignore
    y_data = df_sorted[component_2_col_name].values.reshape(-1, df_sorted[id_col_name].nunique())  # type: ignore
    color_label_data = df_sorted[label_col_name].values.reshape(  # type: ignore
        -1, df_sorted[id_col_name].nunique()
    )  # type: ignore
    size_data = df_sorted[size_col_name].values.reshape(-1, df_sorted[id_col_name].nunique())  # type: ignore

    ids = df_sorted[id_col_name].unique()

    return x_data, y_data, color_label_data, size_data, ids, max_timestamps


def plot_trajectories_with_fading_points(
    df: pd.DataFrame,
    component_1_col_name: str = "component_1",
    component_2_col_name: str = "component_2",
    id_col_name: str = " dw_ek_borger",
    timestamp_col_name: str = "timestamp",
    label_col_name: str = "predicted_label",
    size_col_name: str = "y_pred",
    save: bool = False,
    point_color_legend: dict[int, str] | None = None,
    keep_points: bool = False,
    patients_to_keep: list[int] | None = None,
):
    x_data, y_data, color_label_data, size_data, ids, max_timestamps = _prepare_df_for_trajectories(
        df=df,
        component_1_col_name=component_1_col_name,
        component_2_col_name=component_2_col_name,
        id_col_name=id_col_name,
        timestamp_col_name=timestamp_col_name,
        label_col_name=label_col_name,
        size_col_name=size_col_name,
        patients_to_keep=patients_to_keep,
    )

    # Define a continuous color scale from light gray to black with whitee and sample colors from it according to number of unique ids (use hex codes for colors)
    if len(ids) > 1:
        trace_color_palette = px.colors.sample_colorscale(
            colorscale="viridis", high=0.9, low=0.1, samplepoints=len(ids)
        )

        # Make the colors opaque
        trace_color_palette = [
            color.replace("rgb", "rgba").replace(")", ", 0.25)")  # type: ignore
            for color in trace_color_palette  # type: ignore
        ]

    else:
        trace_color_palette = ["rgba(33, 149, 139, 0.25)"]

    point_color_palette = [
        "#FF0000",
        "#008000",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    point_colors = [[point_color_palette[label] for label in row] for row in color_label_data]

    # Create figure
    fig = go.Figure()

    # Create initial traces for each point and their trails
    traces = []
    for i, point_id in enumerate(ids):
        # Create markers for each point
        traces.append(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(color=point_colors[0][i], size=5 + (0.1 * size_data[0][i])),  # noqa: C408
                showlegend=False,
                name=f"Patient {point_id}",
            )
        )
        # Create lines for the trails
        traces.append(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(  # noqa: C408
                    color=trace_color_palette[i], width=2
                ),  # Adjust alpha value here (0.3 for opacity)
                showlegend=False,
                name=f"Trail {point_id}",
            )
        )

    # Add the traces to the figure
    for trace in traces:
        fig.add_trace(trace)

    # Define animation frames
    frames = []
    for frame in range(max_timestamps):
        frame_data = []
        for i, point_id in enumerate(ids):
            # Update the positions of the dots for each frame
            frame_data.append(
                go.Scatter(
                    x=[x_data[frame, i]],
                    y=[y_data[frame, i]],
                    mode="markers",
                    marker=dict(color=point_colors[frame][i], size=5 + (0.1 * size_data[frame, i])),  # noqa: C408
                    showlegend=False,
                    name=f"Paitent {point_id}",
                    # Add the size data when hovering over a point
                    hovertemplate=f"Patient {point_id}<br>Output probability: {round(size_data[frame, i],2)}",
                )
            )
            if keep_points:
                # Also add previous points to the trail
                frame_data.append(
                    go.Scatter(
                        x=[x_data[f, i] for f in range(frame + 1)],
                        y=[y_data[f, i] for f in range(frame + 1)],
                        mode="markers",
                        marker=dict(  # noqa: C408
                            color=[point_colors[f][i] for f in range(frame + 1)],
                            size=[((size_data[f, i] * 0.1) + 5) for f in range(frame + 1)],
                        ),
                        showlegend=False,
                        name=f"Paitent {point_id}",
                        # Add the size data when hovering over a point
                        hovertemplate=[
                            f"Patient {point_id}<br>Age: {round(size_data[f, i],2)}"
                            for f in range(frame + 1)
                        ],
                    )
                )

            # Update the trails for each point
            trail_x = [x_data[f, i] for f in range(frame + 1)]
            trail_y = [y_data[f, i] for f in range(frame + 1)]
            frame_data.append(
                go.Scatter(
                    x=trail_x,
                    y=trail_y,
                    mode="lines",
                    line=dict(  # noqa: C408
                        color=trace_color_palette[i], width=2
                    ),  # Adjust alpha value here (0.3 for opacity)
                    showlegend=False,
                    name=f"Trail {point_id}",
                )
            )
        frames.append(go.Frame(data=frame_data, name=f"Frame {frame}"))

    # Add frames to the figure
    fig.frames = frames

    # Update layout
    fig.update_layout(
        title="Patient Trajectories over Time",
        xaxis=dict(range=[x_data.min() - 0.1, x_data.max() + 0.1]),  # noqa: C408
        yaxis=dict(range=[y_data.min() - 0.1, y_data.max() + 0.1]),  # noqa: C408
        updatemenus=[
            dict(  # noqa: C408
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None]),  # noqa: C408
                    dict(  # noqa: C408
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    ),
                    # add reset button
                ],
                direction="left",
                pad=dict(r=10, t=87),  # noqa: C408
                showactive=True,
                xanchor="right",
                yanchor="top",
            )
        ],
        showlegend=False,  # Hide legend since each trace has no markers
    )

    # Create slider
    slider = dict(  # noqa: C408
        active=0,
        steps=[],
        currentvalue=dict(font=dict(size=12), prefix="Frame: ", visible=True, xanchor="center"),  # noqa: C408
        pad=dict(t=20),  # noqa: C408
    )
    # Add steps to the slider
    for i, _ in enumerate(fig.frames):
        slider["steps"].append(  # type: ignore
            dict(label=str(i), method="animate", args=[[f"Frame {i}"]])  # noqa: C408
        )

    # Add slider to the layout
    fig.update_layout(sliders=[slider])

    # Add a legend that explains the colors of the points outside the plot
    if point_color_legend is not None:
        for label, color in point_color_legend.items():
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    marker=dict(color=point_color_palette[label], size=10),  # noqa: C408
                    showlegend=True,
                    name=f"{color}",
                )
            )
        fig.update_layout(
            title="Patient Trajectories over Time",
            xaxis=dict(range=[x_data.min() - 0.1, x_data.max() + 0.1]),  # noqa: C408
            yaxis=dict(range=[y_data.min() - 0.1, y_data.max() + 0.1]),  # noqa: C408
            showlegend=True,
        )

        # change theme to white background
        fig.update_layout(template="seaborn")

    if save:
        plotly.offline.plot(fig, filename="persistent_trail_effect.html")

    else:
        fig.show()


if __name__ == "__main__":
    df = prepare_eval_data_for_projections(
        experiment_name="bipolar_model_training_text_feature_lb_200_interval_150",
        predictor_df_name="bipolar_text_feature_set_interval_days_150",
    )

    projection_df = perform_projection(df, projecton_algortithm="pca")

    point_color_legend = {0: "False Positive", 1: "True Positive"}

    # Define the range conditions for component_1 and component_2
    condition_1_min, condition_1_max = -10000, -8500
    condition_2_min, condition_2_max = 0.34, 0.351

    # Function to check if any row in a patient's group satisfies the condition
    def patient_in_range(group):  # type: ignore # noqa: ANN201, ANN001
        return (
            (group["component_1"] >= condition_1_min)
            & (group["component_1"] <= condition_1_max)
            & (group["component_2"] >= condition_2_min)
            & (group["component_2"] <= condition_2_max)
        ).any()

    # Filter the DataFrame by keeping all rows for patients that satisfy the condition
    filtered_df = projection_df.groupby("dw_ek_borger").filter(patient_in_range)

    patients_with_tp_or_fp = filtered_df[filtered_df["prediction_type"].isin(["TP", "FP"])][
        "dw_ek_borger"
    ].unique()

    filtered_df = filtered_df[filtered_df["dw_ek_borger"].isin(patients_with_tp_or_fp)]

    # Step 1: Identify patients with at least one 'TP'
    tp_patients = filtered_df[filtered_df["prediction_type"] == "TP"]["dw_ek_borger"].unique()

    # Step 2: Identify patients with at least one 'FP'
    fp_patients = filtered_df[filtered_df["prediction_type"] == "FP"]["dw_ek_borger"].unique()

    # Step 3: Update 'prediction_type' column for TP patients
    filtered_df.loc[filtered_df["dw_ek_borger"].isin(tp_patients), "prediction_type"] = "TP"

    # Step 4: Update 'prediction_type' column for FP patients (this will overwrite any TP patients with FP if both exist)
    filtered_df.loc[filtered_df["dw_ek_borger"].isin(fp_patients), "prediction_type"] = "FP"

    filtered_df["prediction_type_int"] = filtered_df["prediction_type"].apply(
        lambda x: 1 if x == "TP" else 0
    )

    filtered_df["component_2"] = filtered_df["component_2"] * 100

    ## TSNE ##
    plot_trajectories_with_fading_points(
        filtered_df,
        component_1_col_name="component_1",
        component_2_col_name="component_2",
        id_col_name="dw_ek_borger",
        timestamp_col_name="timestamp",
        label_col_name="prediction_type_int",
        size_col_name="y_hat_prob",
        save=False,
        point_color_legend=point_color_legend,
        keep_points=False,
    )

    patients_to_keep = [1392046]  # 6570800
    patient_df = filtered_df.copy()[filtered_df.copy()["dw_ek_borger"].isin(patients_to_keep)]
    patient_df = patient_df.drop([5292, 51075, 53341])

    plot_trajectories_with_fading_points(
        patient_df,
        component_1_col_name="component_1",
        component_2_col_name="component_2",
        id_col_name="dw_ek_borger",
        timestamp_col_name="timestamp",
        label_col_name="prediction_type_int",
        size_col_name="y_hat_prob",
        save=False,
        point_color_legend=point_color_legend,
        keep_points=False,
    )
    ## TSNE ##
    df = prepare_eval_data_for_projections(
        experiment_name="bipolar_model_training_full_feature_lb_200_interval_150",
        predictor_df_name="bipolar_all_features_interval_days_150",
    )

    projection_df = perform_projection(df, projecton_algortithm="pca")

    # keep only patients with TN and TP
    projection_df = projection_df[projection_df["prediction_type"].isin(["TP", "FP"])]  # type: ignore

    point_color_legend = {0: "True Positive", 1: "False Positive"}

    projection_df["prediction_type_int"] = projection_df["prediction_type"].apply(
        lambda x: 0 if x == "TP" else 1
    )

    patients_to_keep = projection_df["dw_ek_borger"].unique()[:5]
    projection_df = projection_df[projection_df["dw_ek_borger"].isin(patients_to_keep)]

    plot_trajectories_with_fading_points(
        projection_df,
        component_1_col_name="component_1",
        component_2_col_name="component_2",
        id_col_name="dw_ek_borger",
        timestamp_col_name="timestamp",
        label_col_name="prediction_type_int",
        size_col_name="y_hat_prob",
        save=False,
        point_color_legend=point_color_legend,
        keep_points=False,
    )

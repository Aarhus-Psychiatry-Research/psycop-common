from typing import Iterable

import altair as alt
import numpy as np
import pandas as pd

from psycopt2d.visualization.base_charts import plot_bar_chart


def plot_feature_importances(
    val_X_column_names: Iterable[str],
    feature_importances: Iterable[str],
) -> alt.Chart:
    """Plots feature importances.

    Args:
        val_X_column_names (Iterable[str]): Feature names
        feature_importances (np.ndarray): Feature importances

    Returns:
        alt.Chart: Bar chart of feature importances
    """
    sorted_idx = np.array(feature_importances).argsort()

    return plot_bar_chart(
        x_values=pd.Series(val_X_column_names)[sorted_idx],
        y_values=feature_importances[sorted_idx],
        x_title="Feature name",
        y_title="Feature importance",
    )

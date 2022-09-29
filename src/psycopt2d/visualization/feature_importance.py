from pathlib import Path
from typing import Iterable, Optional, Union

import altair as alt
import numpy as np

from psycopt2d.visualization.base_charts import plot_basic_chart


def plot_feature_importances(
    column_names: Iterable[str],
    feature_importances: Union[list[float], np.ndarray],
    top_n_feature_importances: int,
    save_path: Optional[Path] = None,
) -> alt.Chart:
    """Plots feature importances.

    Sklearn's standard feature importance metric is "gain"/information gain,
    which is the decrease in node impurity after the dataset is split on an
    attribute. Node impurity is measured by Gini impurity and is maximal when
    the distribution of the two classes in a node is even, and minimal when the
    classes are perfectly split.

    Args:
        column_names (Iterable[str]): Column/feature names
        feature_importances (Iterable[str]): Feature importances
        top_n_feature_importances (int): Top n features to plot
        save_path (Optional[Path], optional): Path to save the plot. Defaults to None.

    Returns:
        alt.Chart: Horizontal barchart of feature importances
    """

    feature_importances = np.array(feature_importances)
    # argsort sorts in ascending order, need to reverse
    sorted_idx = feature_importances.argsort()[::-1]

    feature_names = np.array(column_names)[sorted_idx][:top_n_feature_importances]
    feature_importances = feature_importances[sorted_idx][:top_n_feature_importances]

    return plot_basic_chart(
        x_values=feature_names,
        y_values=feature_importances,
        x_title="Feature importance (gain)",
        y_title="Feature name",
        sort_x=np.arange(len(feature_importances)),
        plot_type="hbar",
        save_path=save_path,
    )

from typing import Iterable, List, Union

import altair as alt
import numpy as np

from psycopt2d.visualization.base_charts import plot_bar_chart


def plot_feature_importances(
    column_names: Iterable[str],
    feature_importances: Union[List[float], np.ndarray],
    top_n_feature_importances: int,
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

    Returns:
        alt.Chart: Horizontal barchart of feature importances
    """

    feature_importances = np.array(feature_importances)
    # argsort sorts in ascending order, need to reverse
    sorted_idx = feature_importances.argsort()[::-1]

    feature_names = np.array(column_names)[sorted_idx][:top_n_feature_importances]
    feature_importances = feature_importances[sorted_idx][:top_n_feature_importances]

    return plot_bar_chart(
        x_values=feature_importances,
        y_values=feature_names,
        x_title="Feature importance (gain)",
        y_title="Feature name",
        sort_y=np.arange(len(feature_importances)),
    )

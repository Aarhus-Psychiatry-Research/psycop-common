"""Generate feature importances chart."""

from pathlib import Path
from textwrap import wrap
from typing import Optional, Union

import numpy as np
import pandas as pd
from psycop_model_evaluation.base_charts import (
    plot_basic_chart,
)


def plot_feature_importances(
    feature_importance_dict: dict[str, float],
    top_n_feature_importances: int = 10,
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plots feature importances.

    Sklearn's standard feature importance metric is "gain"/information gain,
    which is the decrease in node impurity after the dataset is split on an
    attribute. Node impurity is measured by Gini impurity and is maximal when
    the distribution of the two classes in a node is even, and minimal when the
    classes are perfectly split.

    Args:
        feature_importance_dict (Iterable[str]): Dict with feature_names as keys and importances as values.
        top_n_feature_importances (int): Top n features to plot
        save_path (Optional[Path], optional): Path to save the plot. Defaults to None.

    Returns:
        Union[None, Path]: Path to the saved plot if save_path is not None, else None
    """
    feature_names = list(feature_importance_dict.keys())
    feature_importances = list(feature_importance_dict.values())

    df = pd.DataFrame(
        {"feature_names": feature_names, "feature_importances": feature_importances},
    )
    df = df.sort_values("feature_importances", ascending=False)
    df = df.iloc[:top_n_feature_importances]
    # wrap labels to fit in plot
    df["feature_names"] = df["feature_names"].apply(lambda x: "\n".join(wrap(x, 20)))

    return plot_basic_chart(
        x_values=df["feature_names"],
        y_values=df["feature_importances"],
        x_title="Feature importance (gain)",
        y_title="Feature name",
        sort_x=np.flip(np.arange(len(df["feature_importances"]))),  # type: ignore
        plot_type="hbar",
        fig_size=(8, 5),
        save_path=save_path,
    )

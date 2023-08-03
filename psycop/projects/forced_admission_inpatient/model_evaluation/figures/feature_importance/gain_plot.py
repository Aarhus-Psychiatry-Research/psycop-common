"""Generate feature importances chart."""

from pathlib import Path
from typing import Literal, Union

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    geom_bar,
    geom_text,
    ggplot,
    labs,
    theme,
)

from psycop.projects.forced_admission_inpatient.model_evaluation.config import (
    COLOURS,
    FIGURES_PATH,
    PN_THEME,
)
from psycop.projects.forced_admission_inpatient.model_evaluation.utils.feature_name_to_readable import (
    feature_name_to_readable,
)


def plot_gain(  #
    feature_importance_dict: dict[str, float],
    top_n: int = 10,
    save_path: Path = FIGURES_PATH,
    model: Literal["baseline", "text"] = "baseline",
) -> Union[None, tuple]:
    """Plots feature importances based on gain.

    Sklearn's standard feature importance metric is "gain"/information gain,
    which is the decrease in node impurity after the dataset is split on an
    attribute. Node impurity is measured by Gini impurity and is maximal when
    the distribution of the two classes in a node is even, and minimal when the
    classes are perfectly split.

    Args:
        feature_importance_dict (Iterable[str]): Dict with feature_names as keys and importances as values.
        top_n (int): Top n features to plot
        save_path (Path): Path to save the plot. Defaults to FIGURES_PATH.
        model (Literal["baseline";, "text"], optional):  Name of model. Defaults to "baseline".

    Returns:
        Union[None, Path]: Path to the saved plot if save_path is not None, else None

    Returns:
        Union[None, tuple]: _description_
    """

    # create dir if not exist
    if not Path.exists(save_path):
        save_path.mkdir(parents=True, exist_ok=True)

    # create and filter df to plot
    feature_names = list(feature_importance_dict.keys())
    feature_importances = list(feature_importance_dict.values())
    df = pd.DataFrame(
        {"feature_names": feature_names, "feature_importances": feature_importances},
    )
    df = df.sort_values(by="feature_importances", ascending=False)
    df = df.iloc[:top_n].reset_index(drop=True)
    df = df.sort_values(by="feature_importances", ascending=True)

    # transform feature names to readable feature names
    df["feature_names"] = [
        feature_name_to_readable(feature_name_row)
        for feature_name_row in df["feature_names"]
    ]

    # sort categorical axis
    df["feature_names"] = pd.Categorical(
        df["feature_names"],
        categories=df["feature_names"],
        ordered=True,
    )

    # create plot
    nudge_y = max(df["feature_importances"]) / 30

    p = (
        ggplot(data=df, mapping=aes(x="feature_names", y="feature_importances"))
        + geom_bar(stat="identity", fill=COLOURS["blue"], width=0.9)
        + labs(
            x="",
            y="Information Gain",
            title=f"{model.capitalize()} model: Information gain",
        )
        + geom_text(
            mapping=aes(label="feature_importances"),
            size=8,
            format_string="{:.3f}",
            nudge_y=-nudge_y,
        )
        + coord_flip()
        + PN_THEME
        + theme(figure_size=(7, 4))
    )

    # save plot
    p.save(filename="feature_importances_gain.png", path=str(save_path))

    return (save_path, "feature_importances_gain.png")

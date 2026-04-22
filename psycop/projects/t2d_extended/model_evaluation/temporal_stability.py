"""Main figure for temporal stability evaluation of the T2D-extended model.
Shows AUROC for each test year, with a zoomed inset showing a linear regression of the AUROCs over time.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
import statsmodels.api as sm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def set_plot_style():
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["font.size"] = 11


def fit_wlr(x: np.ndarray, y: np.ndarray, weights: np.ndarray = None):
    X = sm.add_constant(x)
    model = sm.WLS(y, X, weights=weights).fit()
    pred = model.get_prediction(X)

    return {
        "model": model,
        "pred_mean": pred.predicted_mean,
        "pred_ci": pred.conf_int(),
        "slope": model.params[1],
        "slope_se": model.bse[1],
    }


def plot_aurocs(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, xlim: tuple = (2017.7, 2023.3), ylim: tuple = (0, 1)
):
    ax.scatter(x, y, s=60, facecolors="white", edgecolors="black", linewidth=1.2)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)


def add_zoom_box(ax: plt.Axes, zoom_xlim: tuple, zoom_ylim: tuple):
    rect = plt.Rectangle(
        (zoom_xlim[0], zoom_ylim[0]),
        zoom_xlim[1] - zoom_xlim[0],
        zoom_ylim[1] - zoom_ylim[0],
        linewidth=1.0,
        edgecolor="darkgray",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)


def add_inset(
    ax_main: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    lr_result: dict,
    zoom_xlim: tuple,
    zoom_ylim: tuple,
):
    ax_inset = inset_axes(
        ax_main,
        width="35%",
        height="35%",
        loc="lower center",
        bbox_to_anchor=(0, 0.08, 1, 1),
        bbox_transform=ax_main.transAxes,
        borderpad=1.2,
    )

    ax_inset.scatter(x, y, s=35, facecolors="white", edgecolors="black", linewidth=1.0)

    if lr_result is not None:
        ax_inset.plot(x, lr_result["pred_mean"], color="darkblue", linewidth=1.2)
        ax_inset.fill_between(
            x, lr_result["pred_ci"][:, 0], lr_result["pred_ci"][:, 1], color="darkblue", alpha=0.07
        )

    ax_inset.set_xlim(*zoom_xlim)
    ax_inset.set_ylim(*zoom_ylim)
    ax_inset.tick_params(axis="both", labelsize=8)

    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.8)

    ax_inset.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    mark_inset(
        ax_main, ax_inset, loc1=2, loc2=4, fc="none", ec="darkgray", linewidth=0.8, alpha=0.5
    )

    return ax_inset


def plot_aurocs_with_inset(
    auroc_dict: dict,
    weights: np.ndarray = None,
    zoom_xlim: tuple = (2017.7, 2023.3),
    zoom_ylim: tuple = (0.75, 0.91),
    show_regression: bool = True,
    exp_path=None,
):
    set_plot_style()

    years = np.array(sorted(auroc_dict.keys()))
    x = 2000 + years
    y = np.array([auroc_dict[k] for k in years])

    lr_result = (
        fit_wlr(x, y, weights=np.array([n_contacts[k] for k in years])) if show_regression else None
    )

    if lr_result:
        print(f"Slope:          {lr_result['slope']:.6f}")
        print(f"Standard error: {lr_result['slope_se']:.6f}")

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

    plot_aurocs(ax, x, y, xlim=(2017.7, 2023.3))  # main: no regression line
    add_zoom_box(ax, zoom_xlim, zoom_ylim)
    add_inset(ax, x, y, lr_result, zoom_xlim, zoom_ylim)  # inset: with regression line

    plt.tight_layout()

    fig.savefig("auroc_linear-regression_inset.png")

    if exp_path is not None:
        fig.savefig(Path(exp_path) / "auroc_linear-regression_inset.png")

    plt.show()


if __name__ == "__main__":
    exp_path = "E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/"

    auroc_dict = {
        18: 0.8376074504993402,
        19: 0.8331785757189495,
        20: 0.8716330715094136,
        21: 0.7956406780790461,
        22: 0.8399708928832673,
        23: 0.7985506465746504,
    }

    n_contacts = {18: 210424, 19: 186669, 20: 157332, 21: 150688, 22: 163294, 23: 42674}

    plot_aurocs_with_inset(auroc_dict, weights=n_contacts, exp_path=exp_path)

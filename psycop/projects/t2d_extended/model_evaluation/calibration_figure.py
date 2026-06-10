"""Generate calibration plots for each test year of the T2D-extended project.
Produces a panel of calibration curves with confidence intervals, along with histograms of predicted probabilities."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binom
from sklearn.metrics import brier_score_loss

from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.projects.t2d_extended.model_evaluation.utils.parse_utils import eval_df_to_eval_dataset


def logit(p: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    brier = brier_score_loss(y_true, y_pred)

    lp = logit(y_pred)
    X = sm.add_constant(lp)
    model = sm.Logit(y_true, X).fit(disp=0)

    intercept = model.params[0]
    slope = model.params[1]

    return intercept, slope, brier


def compute_bin_stats(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-8
    bins[-1] += 1e-8

    bin_centers = []
    observed = []
    lower_ci = []
    upper_ci = []
    counts = []

    for i in range(n_bins):
        mask = (y_pred > bins[i]) & (y_pred <= bins[i + 1])
        n = mask.sum()
        if n == 0:
            continue

        y_bin = y_true[mask]
        p_bin = y_pred[mask].mean()
        obs = y_bin.mean()

        ci_low, ci_up = binom.interval(0.95, n, obs, loc=0)
        ci_low /= n
        ci_up /= n

        bin_centers.append(p_bin)
        observed.append(obs)
        lower_ci.append(ci_low)
        upper_ci.append(ci_up)
        counts.append(n)

    return (
        np.array(bin_centers),
        np.array(observed),
        np.array(lower_ci),
        np.array(upper_ci),
        np.array(counts),
    )


def load_year_data(
    year: int, training_start_date: str, training_end_date: str, exp_path: str
) -> tuple[np.ndarray, np.ndarray]:
    eval_start = f"{year}-01-01"
    eval_end = f"{year}-12-31"

    run_path = f"{exp_path}" f"{training_start_date}_{training_end_date}_{eval_start}_{eval_end}"

    cfg = PsycopConfig().from_disk(f"{run_path}/config.cfg")
    eval_dataset = eval_df_to_eval_dataset(cfg)

    y_true = np.asarray(eval_dataset.y)
    y_pred = np.asarray(eval_dataset.y_hat_probs)

    return y_true, y_pred


def compute_year_metrics(year: int, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    intercept, slope, brier = calibration_metrics(y_true, y_pred)

    return {
        "year": year,
        "intercept": intercept,
        "slope": slope,
        "brier": brier,
        "n": len(y_true),
        "event_rate": y_true.mean(),
    }


def plot_calibration_panel(
    ax_curve: plt.Axes,
    ax_hist: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    year: int,
    xmax: float,
    ymax: float,
    show_legend: bool = False,
) -> None:
    bin_centers, obs, ci_low, ci_up, _ = compute_bin_stats(y_true, y_pred, n_bins=10)

    ax_curve.plot(bin_centers, obs, marker="o", linewidth=1.8, label="Model calibration")
    ax_curve.fill_between(bin_centers, ci_low, ci_up, alpha=0.2)

    ax_curve.plot(
        [0, xmax],
        [0, xmax],
        linestyle="--",
        color="gray",
        linewidth=1.2,
        label="Perfect calibration",
    )

    ax_curve.set_title(str(year))
    ax_curve.set_xlim(0, xmax)
    ax_curve.set_ylim(0, ymax)
    ax_curve.set_ylabel("Observed frequency")

    if show_legend:
        ax_curve.legend(loc="lower right", frameon=True)

    # Histogram
    counts, edges = np.histogram(y_pred, bins=25, range=(0, xmax))
    counts[counts < 5] = 0

    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    ax_hist.bar(centers, counts, width=width, color="gray", alpha=0.35, edgecolor="none")
    ax_hist.set_xlim(0, xmax)
    ax_hist.set_ylabel("Count")

    ax_hist.sharex(ax_curve)

    # Style
    for ax in (ax_curve, ax_hist):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def calibration_plot(
    training_start_date: str, training_end_date: str, test_years: list, exp_path: str
) -> pd.DataFrame:
    xmax = 0.025
    ymax = 0.035

    n_years = len(test_years)
    n_cols = math.ceil(math.sqrt(n_years))
    n_row_blocks = math.ceil(n_years / n_cols)
    n_rows = n_row_blocks * 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_row_blocks * 2))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    rows = []

    for i, year in enumerate(test_years):
        y_true, y_pred = load_year_data(year, training_start_date, training_end_date, exp_path)

        rows.append(compute_year_metrics(year, y_true, y_pred))

        row = (i // n_cols) * 2
        col = i % n_cols

        plot_calibration_panel(
            axes[row, col],
            axes[row + 1, col],
            y_true,
            y_pred,
            year,
            xmax,
            ymax,
            show_legend=(i == 0),
        )

        print(
            f"{year}: n={len(y_pred)}, "
            f"mean={y_pred.mean():.6f}, std={y_pred.std():.6f}, "
            f"min={y_pred.min():.6f}, max={y_pred.max():.6f}, "
            f"event_rate={y_true.mean():.6f}"
        )

    # Hide unused axes
    total_slots = n_row_blocks * n_cols
    for j in range(n_years, total_slots):
        row = (j // n_cols) * 2
        col = j % n_cols
        axes[row, col].axis("off")
        axes[row + 1, col].axis("off")

    df = pd.DataFrame(rows)
    print(df.round(6))

    plt.tight_layout()
    plt.savefig(f"{exp_path}/calibration_temporal_validation_raw.png", dpi=300, bbox_inches="tight")
    plt.show()

    return df


if __name__ == "__main__":
    training_start_date = "2013-01-01"
    training_end_date = "2018-01-01"
    test_years = [2018, 2019, 2020, 2021, 2022, 2023]
    exp_path = "psycop"

    calibration_plot(
        training_start_date=training_start_date,
        training_end_date=training_end_date,
        test_years=test_years,
        exp_path=exp_path,
    )

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.projects.t2d_extended.model_training.USING_temporal_evaluation import eval_df_to_eval_dataset



# fig, axes = plt.subplots(2,3, figsize=(12,8))

# for i,year in enumerate(years):
    
#     ax = axes.flatten()[i]
    
#     y_true = y_test_by_year[year]
#     y_pred = pred_prob_by_year[year]
    
#     prob_true, prob_pred = calibration_curve(
#         y_true, y_pred, n_bins=10
#     )

#     ax.plot(prob_pred, prob_true, marker='o', label="Model")
#     ax.plot([0,1],[0,1], linestyle='--', color='gray', label="Perfect")

#     ax.set_title(str(year))
#     ax.set_xlabel("Predicted probability")
#     ax.set_ylabel("Observed frequency")
#     ax.set_xlim(0,1)
#     ax.set_ylim(0,1)

# plt.tight_layout()
# plt.show()

# BEST_POS_RATE = 0.03

# def calibration_plot(cfg: PsycopConfig):
#     eval_dataset = eval_df_to_eval_dataset(cfg)
    
#     df = pd.DataFrame(
#         {
#             "true": eval_dataset.y,
#             "pred": eval_dataset.get_predictions_for_positive_rate(BEST_POS_RATE)[0],
#         }
#     )

#     # Calculate calibration curve
#     prob_true, prob_pred = calibration_curve(
#         df.true, df.pred, n_bins=10
#     )



# if __name__ == "__main__":

#     training_start_date="2013-01-01"
#     training_end_date="2018-01-01"
#     evaluation_interval=(f"2018-01-01", f"2018-12-31")

#     run_path = f"training/T2D-extended, temporal validation/{training_start_date}_{training_end_date}_{evaluation_interval[0]}_{evaluation_interval[1]}"

#     cfg = PsycopConfig().from_disk(f"{run_path}/config.cfg")

#     calibration = calibration_plot(cfg)

#     print("hey")



# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.calibration import calibration_curve

# BEST_POS_RATE = 0.03
# TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]

# def calibration_plot_grid(training_start_date, training_end_date):
#     """
#     Create a 2x3 grid of calibration plots for each test year.
#     """
#     fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
#     for i, year in enumerate(TEST_YEARS):
#         eval_start = f"{year}-01-01"
#         eval_end = f"{year}-12-31"
        
#         # Build run path for this evaluation year
#         run_path = f"E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/{training_start_date}_{training_end_date}_{eval_start}_{eval_end}"
#         cfg = PsycopConfig().from_disk(f"{run_path}/config.cfg")
        
#         # Load dataset and predictions
#         eval_dataset = eval_df_to_eval_dataset(cfg)
#         df = pd.DataFrame({
#             "true": eval_dataset.y,
#             "pred": eval_dataset.get_predictions_for_positive_rate(BEST_POS_RATE)[0]
#         })
        
#         # Compute calibration curve
#         prob_true, prob_pred = calibration_curve(df.true, df.pred, n_bins=10)
        
#         # Plot
#         ax = axes.flatten()[i]
#         ax.plot(prob_pred, prob_true, marker='o', label="Model")
#         ax.plot([0,1], [0,1], linestyle='--', color='gray', label="Perfect")
#         ax.set_title(str(year))
#         ax.set_xlabel("Predicted probability")
#         ax.set_ylabel("Observed frequency")
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         ax.legend(loc="lower right")
    
#     plt.tight_layout()
#     plt.suptitle("Calibration Plots by Test Year", fontsize=16, y=1.02)
#     plt.savefig("E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/calibration_temporal_validation.png", dpi=300, bbox_inches="tight")

#     plt.show()

#     print("hey")


# if __name__ == "__main__":
#     training_start_date = "2013-01-01"
#     training_end_date = "2018-01-01"
    
#     calibration_plot_grid(training_start_date, training_end_date)
#     print("Calibration plots done!")




# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.calibration import calibration_curve
# from scipy.stats import binom

# # BEST_POS_RATE = 0.03
# TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]



# import numpy as np
# import pandas as pd
# from sklearn.metrics import brier_score_loss
# import statsmodels.api as sm


# def logit(p):
#     eps = 1e-6
#     p = np.clip(p, eps, 1 - eps)
#     return np.log(p / (1 - p))


# def calibration_metrics(y_true, y_pred):

#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)

#     # ----- Brier -----
#     brier = brier_score_loss(y_true, y_pred)

#     # ----- slope & intercept -----
#     lp = logit(y_pred)

#     X = sm.add_constant(lp)  # intercept + slope
#     model = sm.Logit(y_true, X).fit(disp=0)

#     intercept = model.params[0]
#     slope = model.params[1]

#     return intercept, slope, brier

# def compute_bin_stats(y_true, y_pred, n_bins=10):
#     """
#     Computes binned calibration statistics with quantile bins
#     and binomial confidence intervals.
#     """
#     # Quantile bins
#     bins = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
#     bins[0] -= 1e-8  # include 0
#     bins[-1] += 1e-8  # include 1
    
#     bin_centers = []
#     observed = []
#     lower_ci = []
#     upper_ci = []
#     counts = []
    
#     for i in range(n_bins):
#         mask = (y_pred > bins[i]) & (y_pred <= bins[i+1])
#         n = mask.sum()
#         if n == 0:
#             continue
#         y_bin = y_true[mask]
#         p_bin = y_pred[mask].mean()
#         obs = y_bin.mean()
#         # Binomial CI (95%)
#         ci_low, ci_up = binom.interval(0.95, n, obs, loc=0)
#         ci_low /= n
#         ci_up /= n
        
#         bin_centers.append(p_bin)
#         observed.append(obs)
#         lower_ci.append(ci_low)
#         upper_ci.append(ci_up)
#         counts.append(n)
        
#     return np.array(bin_centers), np.array(observed), np.array(lower_ci), np.array(upper_ci), np.array(counts)

# def calibration_plot_grid_pub(training_start_date, training_end_date):
#     fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
#     rows = []

#     for i, year in enumerate(TEST_YEARS):
#         eval_start = f"{year}-01-01"
#         eval_end = f"{year}-12-31"
        
#         run_path = f"E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/{training_start_date}_{training_end_date}_{eval_start}_{eval_end}"
#         cfg = PsycopConfig().from_disk(f"{run_path}/config.cfg")
        
#         eval_dataset = eval_df_to_eval_dataset(cfg)
#         y_true = eval_dataset.y
#         y_pred = eval_dataset.y_hat_probs
        
#         intercept, slope, brier = calibration_metrics(y_true, y_pred)

#         rows.append(
#                 {
#                     "year": year,
#                     "intercept": intercept,
#                     "slope": slope,
#                     "brier": brier,
#                     "n": len(y_true),
#                     "event_rate": y_true.mean(),
#                 }
#             )

#         # Compute binned stats
#         bin_centers, obs, ci_low, ci_up, counts = compute_bin_stats(y_true, y_pred, n_bins=10)

#         ax = axes.flatten()[i]
#         # Plot model calibration with shaded CI
#         ax.plot(bin_centers, obs, marker="o", color="tab:blue", label="Model")
#         ax.fill_between(bin_centers, ci_low, ci_up, color="tab:blue", alpha=0.2)
#         # Perfect calibration line
#         ax.plot([0, 0.1], [0, 0.1], linestyle="--", color="gray")
                
#         # Histogram of predicted probabilities
#         ax_hist = ax.twinx()
#         ax_hist.hist(y_pred, bins=20, color="gray", alpha=0.1)
#         ax_hist.set_yticks([])
        
#                 # Small bar plot below showing patient count per bin
#         for j, center in enumerate(bin_centers):
#             ax.bar(center, 0.03, bottom=-0.03, width=0.03, color="gray", alpha=0.5)  # scaled small bars
#             ax.text(center, -0.05, str(counts[j]), ha='center', va='top', fontsize=8)
        

#         ax.set_title(str(year))
#         ax.set_xlabel("Predicted probability")
#         ax.set_ylabel("Observed frequency")
#         ax.set_xlim(0, 0.1)
#         ax.set_ylim(0, 0.05)
#         ax.legend(loc="lower right")
    
#     df = pd.DataFrame(rows)

#     print(df)

#     plt.tight_layout()
#     plt.suptitle("Calibration Plots by Test Year (Quantile Bins + CI + Histogram)", fontsize=16, y=1.02)
    
#     # Save figure
#     plt.savefig("E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/calibration_temporal_validation_pub_raw.png", dpi=300, bbox_inches="tight")
#     plt.savefig("E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/calibration_temporal_validation_pub_raw.pdf", bbox_inches="tight")
    
#     plt.show()


# if __name__ == "__main__":
#     training_start_date = "2013-01-01"
#     training_end_date = "2018-01-01"

#     calibration_plot_grid_pub(training_start_date, training_end_date)
#     print("Publication-ready calibration plots saved!")





import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import binom
from sklearn.metrics import brier_score_loss
import statsmodels.api as sm

TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]


def logit(p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    brier = brier_score_loss(y_true, y_pred)

    lp = logit(y_pred)
    X = sm.add_constant(lp)
    model = sm.Logit(y_true, X).fit(disp=0)

    intercept = model.params[0]
    slope = model.params[1]

    return intercept, slope, brier


def compute_bin_stats(y_true, y_pred, n_bins=10):
    """
    Compute quantile-binned calibration statistics with 95% binomial CI.
    """
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


def calibration_plot_grid_pub(training_start_date, training_end_date):
    rows = []

    # Tighter zoom for rare outcome setting
    xmax = 0.025
    ymax = 0.035

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(
        4,
        3,
        height_ratios=[3, 0.7, 3, 0.7],
        hspace=0.45,
        wspace=0.25,
    )

    for i, year in enumerate(TEST_YEARS):
        eval_start = f"{year}-01-01"
        eval_end = f"{year}-12-31"

        run_path = (
            f"E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/"
            f"{training_start_date}_{training_end_date}_{eval_start}_{eval_end}"
        )
        cfg = PsycopConfig().from_disk(f"{run_path}/config.cfg")

        eval_dataset = eval_df_to_eval_dataset(cfg)
        y_true = np.asarray(eval_dataset.y)
        y_pred = np.asarray(eval_dataset.y_hat_probs)

        intercept, slope, brier = calibration_metrics(y_true, y_pred)
        rows.append(
            {
                "year": year,
                "intercept": intercept,
                "slope": slope,
                "brier": brier,
                "n": len(y_true),
                "event_rate": y_true.mean(),
            }
        )

        bin_centers, obs, ci_low, ci_up, counts = compute_bin_stats(
            y_true, y_pred, n_bins=10
        )

        row = (i // 3) * 2
        col = i % 3

        ax_curve = fig.add_subplot(gs[row, col])
        ax_hist = fig.add_subplot(gs[row + 1, col], sharex=ax_curve)

        # ---- calibration curve ----
        ax_curve.plot(
            bin_centers,
            obs,
            marker="o",
            linewidth=1.8,
            label="Model calibration",
        )
        ax_curve.fill_between(
            bin_centers,
            ci_low,
            ci_up,
            alpha=0.2,
        )
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
        ax_curve.set_xticks([0.00, 0.01, 0.02, 0.03])
        ax_curve.set_yticks([0.00, 0.01, 0.02, 0.03])
        ax_curve.legend(loc="lower right", frameon=True)

        # ---- histogram underneath ----
        counts, edges = np.histogram(
            y_pred,
            bins=25,
            range=(0, xmax),
        )

        # privacy filter
        counts[counts < 5] = 0

        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        ax_hist.bar(
            centers,
            counts,
            width=width,
            color="gray",
            alpha=0.35,
            edgecolor="none",
        )
        ax_hist.set_xlim(0, xmax)
        if row == 2:  # only bottom block
            ax_hist.set_xlabel("Predicted probability")
        else:
            ax_hist.set_xlabel("")
        ax_hist.set_ylabel("Count")
        ax_hist.set_xticks([0.00, 0.01, 0.02, 0.03])

        # Cleaner look
        ax_curve.spines["top"].set_visible(False)
        ax_curve.spines["right"].set_visible(False)
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)

        print(
        f"{year}: n={len(y_pred)}, "
        f"mean={y_pred.mean():.6f}, std={y_pred.std():.6f}, "
        f"min={y_pred.min():.6f}, max={y_pred.max():.6f}, "
        f"event_rate={y_true.mean():.6f}"
    )


    df = pd.DataFrame(rows)
    print(df.round(6))


    fig.suptitle(
        "",
        fontsize=16,
        y=0.98,
    )

    save_base = "E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/calibration_temporal_validation_raw"
    plt.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    training_start_date = "2013-01-01"
    training_end_date = "2018-01-01"

    calibration_plot_grid_pub(training_start_date, training_end_date)
    print("Publication-ready calibration plots saved!")
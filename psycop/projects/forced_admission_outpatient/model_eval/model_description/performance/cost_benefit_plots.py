import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
import patchworklib as pw
import plotnine as pn
from scipy.stats import truncnorm

from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.forced_admission_outpatient.model_eval.config import COLORS, FA_PN_THEME
from psycop.projects.forced_admission_outpatient.model_eval.model_description.performance.performance_by_ppr import (
    _get_num_of_unique_outcome_events,  # type: ignore
    _get_number_of_outcome_events_with_at_least_one_true_positve,  # type: ignore
)
from psycop.projects.forced_admission_outpatient.utils.pipeline_objects import (
    ForcedAdmissionOutpatientPipelineRun,
)


def _sample_float_from_truncated_log_normal(
    mean_efficiency: float, lower_bound: float, upper_bound: float, n: int = 10000
) -> np.ndarray:  # type: ignore
    # Log-normal distribution parameters
    # Calculate the parameters of the underlying normal distribution
    # Mean (mu) and standard deviation (sigma) of the underlying normal distribution
    sigma = 0.5  # You might need to adjust this to get the exact mean and skewness
    mu = np.log(mean_efficiency) - 0.5 * sigma**2

    # Define the truncation bounds in terms of the standard normal distribution
    a, b = (np.log(lower_bound) - mu) / sigma, (np.log(upper_bound) - mu) / sigma

    # Sample from the truncated normal distribution
    truncated_normal = truncnorm(a, b, loc=mu, scale=sigma)

    return np.exp(truncated_normal.rvs(size=n))


def _sample_int_from_truncated_normal(
    mean_cost: int, std_cost: int, lower_bound: int, n: int = 10000
) -> np.ndarray:  # type: ignore
    # Calculate the truncation bounds in terms of the standard normal distribution
    a, b = (lower_bound - mean_cost) / std_cost, np.inf

    # Create the truncated normal distribution
    truncated_normal = truncnorm(a, b, loc=mean_cost, scale=std_cost)

    return truncated_normal.rvs(size=n).astype(int)


def sample_cost_benefit_estimates(n: int = 10000) -> pd.DataFrame:
    # sample intervention cost 100 times - a integer from a normal distribution with mean 500 and std 100 and a lower bound of 0
    cost_of_intervention = _sample_int_from_truncated_normal(
        mean_cost=500, std_cost=200, lower_bound=200, n=n
    )

    # sample efficiency of intervention - a float from a log-normal distribution with mean 0.5 and bounds 0.1 and 0.9
    efficiency_of_intervention = _sample_float_from_truncated_log_normal(
        mean_efficiency=0.15, lower_bound=0.01, upper_bound=0.99, n=n
    )

    # sample savings from prevented outcome - a integer from a normal distribution with mean 100000 and std 25000
    savings_from_prevented_outcome = _sample_int_from_truncated_normal(
        mean_cost=100000, std_cost=25000, lower_bound=10000, n=n
    )

    df = pd.DataFrame(
        {
            "cost_of_intervention": cost_of_intervention,
            "efficiency_of_intervention": efficiency_of_intervention,
            "savings_from_prevented_outcome": savings_from_prevented_outcome,
            "cost_benefit_ratio": savings_from_prevented_outcome
            * efficiency_of_intervention
            / cost_of_intervention,
        }
    )

    return df


def plot_sampling_distribution(df: pd.DataFrame, col_to_plot: str, save: bool = False) -> pn.ggplot:
    # Create string for plottign for col name by removing underscores and capitalizing
    col_to_plot_str = col_to_plot.replace("_", " ").capitalize()

    p = (
        pn.ggplot(df, pn.aes(x=col_to_plot))
        + pn.geom_histogram(
            pn.aes(y=pn.after_stat("density")),
            bins=30,
            fill=COLORS.secondary,
            alpha=0.4,
            color=COLORS.secondary,
        )
        + pn.geom_density(color=COLORS.tertiary, size=1.5)
        + pn.labs(x=f"{col_to_plot_str}", y="Density")
        + FA_PN_THEME
        + pn.theme(panel_grid_major=pn.element_blank(), panel_grid_minor=pn.element_blank())
    )
    # if col_to_plot == "cost_benefit_ratio" then add vertical lines for the 5th, and 95th percentiles, the mean and the median
    if col_to_plot == "cost_benefit_ratio":
        p += pn.geom_vline(
            xintercept=int(df["cost_benefit_ratio"].median()), linetype="dotted", color="red"
        )
        p += pn.annotate(
            "text",
            x=int(df["cost_benefit_ratio"].median()),
            y=0.028,
            label="Median",
            color="red",
            va="center",
            ha="right",
            size=6,
            angle=90,
        )
        p += pn.geom_vline(
            xintercept=int(df["cost_benefit_ratio"].mean()), linetype="dotted", color="blue"
        )
        p += pn.annotate(
            "text",
            x=int(df["cost_benefit_ratio"].mean()),
            y=0.028,
            label="Mean",
            color="blue",
            va="center",
            ha="right",
            size=6,
            angle=90,
        )
        p += pn.geom_vline(
            xintercept=int(np.percentile(df["cost_benefit_ratio"], 5)),
            linetype="dotted",
            color="green",
        )
        p += pn.annotate(
            "text",
            x=int(np.percentile(df["cost_benefit_ratio"], 5)),
            y=0.028,
            label="5th perc.",
            color="green",
            va="center",
            ha="right",
            size=6,
            angle=90,
        )
        p += pn.geom_vline(
            xintercept=int(np.percentile(df["cost_benefit_ratio"], 95)),
            linetype="dotted",
            color="green",
        )
        p += pn.annotate(
            "text",
            x=int(np.percentile(df["cost_benefit_ratio"], 95)),
            y=0.028,
            label="95th perc.",
            color="green",
            va="center",
            ha="right",
            size=6,
            angle=90,
        )

    if save:
        p.save(filename=f"sampling_distribution_{col_to_plot}.png", width=7, height=7)

    return p


def calculate_cost_benefit_estimates_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate the required statistics
    median_value = int(df["cost_benefit_ratio"].median())
    mean_value = int(df["cost_benefit_ratio"].mean())
    percentile_5th = int(np.percentile(df["cost_benefit_ratio"], 5))
    percentile_95th = int(np.percentile(df["cost_benefit_ratio"], 95))

    # Add the statistics to a dictionary
    stats = {
        f"Median ({median_value}:1)": median_value,
        f"Mean ({mean_value}:1)": mean_value,
        f"5th percentile ({percentile_5th}:1)": percentile_5th,
        f"95th percentile ({percentile_95th}:1)": percentile_95th,
    }

    return pd.DataFrame(stats, index=[0])


def calculate_cost_benefit(
    run: ForcedAdmissionOutpatientPipelineRun,
    cost_benefit_ratio: int,
    per_true_positive: bool = True,
    min_alert_days: None | int = 30,
    positive_rates: Sequence[float] = [0.5, 0.2, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01],
) -> pd.DataFrame:
    eval_dataset = run.pipeline_outputs.get_eval_dataset()

    df: pd.DataFrame = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=eval_dataset, positive_rates=positive_rates
    )

    df["Total number of unique outcome events"] = _get_num_of_unique_outcome_events(
        eval_dataset=eval_dataset
    )

    df["Number of positive outcomes in test set (TP+FN)"] = (
        df["true_positives"] + df["false_negatives"]
    )

    df["Number of unique outcome events detected ≥1"] = [
        _get_number_of_outcome_events_with_at_least_one_true_positve(
            eval_dataset=eval_dataset, positive_rate=pos_rate, min_alert_days=min_alert_days
        )
        for pos_rate in positive_rates
    ]

    if per_true_positive:
        df["benefit_harm"] = (
            (df["true_positives"] * cost_benefit_ratio) - (df["false_positives"])
        ) / df["Number of positive outcomes in test set (TP+FN)"]

    else:
        df["benefit_harm"] = (
            (df["Number of unique outcome events detected ≥1"] * cost_benefit_ratio)
            - (df["false_positives"])
        ) / df["Total number of unique outcome events"]

    return df[["benefit_harm", "positive_rate"]]


def plot_cost_benefit_by_ppr(df: pd.DataFrame, per_true_positive: bool) -> pn.ggplot:
    legend_order = sorted(
        df["cost_benefit_ratio_str"].unique(),
        key=lambda s: int(re.sub(r"[^\d]+", "", s.split(":")[0])),
        reverse=True,
    )

    df = df.assign(values=pd.Categorical(df["cost_benefit_ratio_str"], legend_order))

    p = (
        pn.ggplot(df, pn.aes(x="positive_rate", y="benefit_harm", color="cost_benefit_ratio_str"))
        + pn.geom_point()
        + pn.labs(x="Predicted Positive Rate", y="Benefit/Harm")
        + FA_PN_THEME
        + pn.theme(axis_text_x=pn.element_text(rotation=45, hjust=1))
        + pn.labs(color="Profit/Cost ratio")
        + pn.theme(
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            legend_position=(0.4, 0.18),
        )
        + pn.geom_hline(yintercept=0, linetype="dotted", color="red")
        + pn.annotate(
            "rect",
            xmin=float("-inf"),
            xmax=float("inf"),
            ymin=float("-inf"),
            ymax=0,
            fill="red",
            alpha=0.1,
        )
        + pn.annotate(
            "rect",
            xmin=float("-inf"),
            xmax=float("inf"),
            ymin=0,
            ymax=float("inf"),
            fill="green",
            alpha=0.1,
        )
    )

    if per_true_positive:
        p += pn.ggtitle("Cost/benefit pr. positive outcomes")
    else:
        p += pn.ggtitle("Cost/benefit pr. unique outcomes")

    for value in legend_order:
        p += pn.geom_path(df[df["cost_benefit_ratio_str"] == value], group=1)  # type: ignore
    return p


def fa_cost_benefit_by_ratio_and_ppr(
    run: ForcedAdmissionOutpatientPipelineRun,
    per_true_positive: bool,
    cost_benefit_ratios: Sequence[int],
    min_alert_days: None | int = 30,
    positive_rates: Sequence[float] = [0.5, 0.2, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01],
) -> pn.ggplot:
    dfs = []

    for ratio in cost_benefit_ratios:
        df = calculate_cost_benefit(
            run=run,
            cost_benefit_ratio=ratio,
            min_alert_days=min_alert_days,
            per_true_positive=per_true_positive,
            positive_rates=positive_rates,
        )

        cost_benefit_ratio_str = f"{ratio}:1"
        # Convert to string to allow distinct scales for color
        df["cost_benefit_ratio_str"] = str(cost_benefit_ratio_str)
        dfs.append(df)

    plot_df = pd.concat(dfs)

    p = plot_cost_benefit_by_ppr(plot_df, per_true_positive)

    if per_true_positive:
        p.save(
            filename=run.paper_outputs.paths.figures
            / "fa_outpatient_cost_benefit_estimates_per_true_positives.png",
            width=7,
            height=7,
        )
    else:
        p.save(
            filename=run.paper_outputs.paths.figures
            / "fa_outpatient_cost_benefit_estimates_per_true_unique_outcomes.png",
            width=7,
            height=7,
        )

    return p


def fa_cost_benefit_from_monte_carlo_simulations(
    run: ForcedAdmissionOutpatientPipelineRun,
    per_true_positive: bool,
    min_alert_days: None | int = 30,
    positive_rates: Sequence[float] = [0.5, 0.2, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01],
    n: int = 10000,
) -> pn.ggplot:
    df = sample_cost_benefit_estimates(n=n)

    dist_plots = fa_patchwork_sampling_distribution_plots(
        run=run, df=df, cols_to_plot=list(df.columns), grid_plot=False, save=False
    )

    stats = calculate_cost_benefit_estimates_stats(df)

    dfs = []

    for key, item in stats.items():
        df = calculate_cost_benefit(
            run=run,
            cost_benefit_ratio=item[0],
            per_true_positive=per_true_positive,
            min_alert_days=min_alert_days,
            positive_rates=positive_rates,
        )

        # Convert to string to allow distinct scales for color
        df["cost_benefit_ratio_str"] = str(key)
        dfs.append(df)

    plot_df = pd.concat(dfs)

    p = plot_cost_benefit_by_ppr(plot_df, per_true_positive)

    dist_plots.append(p)  # type: ignore

    # add distribution plots to the cost benefit plot
    grid = create_patchwork_grid(plots=dist_plots, single_plot_dimensions=(5.1, 5.1), n_in_row=2)  # type: ignore

    if per_true_positive:
        grid_output_path = (
            run.paper_outputs.paths.figures
            / "fa_outpatient_mc_cost_benefit_estimates_per_positives_outcomes.png"
        )
        grid.savefig(grid_output_path)
    else:
        grid_output_path = (
            run.paper_outputs.paths.figures
            / "fa_outpatient_mc_cost_benefit_estimates_per_unique_outcomes.png"
        )
        grid.savefig(grid_output_path)
    return p


def fa_patchwork_sampling_distribution_plots(
    run: ForcedAdmissionOutpatientPipelineRun,
    df: pd.DataFrame,
    cols_to_plot: Sequence[str],
    grid_plot: bool = False,
    save: bool = False,
) -> pw.Bricks | Sequence[pn.ggplot]:
    plots = [plot_sampling_distribution(df, col_to_plot) for col_to_plot in cols_to_plot]

    if grid_plot:
        grid = create_patchwork_grid(plots=plots, single_plot_dimensions=(5, 5), n_in_row=4)

        if save:
            grid_output_path = (
                run.paper_outputs.paths.figures
                / "fa_outpatient_cost_benefit_intervention_efficiency_sampled_distributions.png"
            )
            grid.savefig(grid_output_path)

        return grid

    return plots


if __name__ == "__main__":
    from psycop.projects.forced_admission_outpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    fa_cost_benefit_from_monte_carlo_simulations(
        run=get_best_eval_pipeline(), per_true_positive=False, min_alert_days=30
    )

    fa_cost_benefit_by_ratio_and_ppr(
        run=get_best_eval_pipeline(),
        per_true_positive=False,
        min_alert_days=30,
        cost_benefit_ratios=[40, 20, 10, 6, 3],
    )

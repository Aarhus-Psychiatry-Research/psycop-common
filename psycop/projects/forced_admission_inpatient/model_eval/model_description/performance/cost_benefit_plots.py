from collections.abc import Sequence

import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from psycop.projects.forced_admission_inpatient.model_eval.config import FA_PN_THEME
from psycop.projects.forced_admission_inpatient.model_eval.model_description.performance.performance_by_ppr import (
    _get_num_of_unique_outcome_events,  # type: ignore
    _get_number_of_outcome_events_with_at_least_one_true_positve,  # type: ignore
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def calculate_cost_benefit(
    run: ForcedAdmissionInpatientPipelineRun,
    savings_from_prevented_outcome: int,
    cost_of_intervention: int,
    efficiency_of_intervention: float,
    per_true_positive: bool = True,
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
            eval_dataset=eval_dataset, positive_rate=pos_rate
        )
        for pos_rate in positive_rates
    ]

    saving_per_positive_pred = savings_from_prevented_outcome * efficiency_of_intervention

    if per_true_positive:
        df["benefit_harm"] = (
            (df["true_positives"] * saving_per_positive_pred)
            - (df["false_positives"] * cost_of_intervention)
        ) / df["Number of positive outcomes in test set (TP+FN)"]

    else:
        df["benefit_harm"] = (
            (df["Number of unique outcome events detected ≥1"] * saving_per_positive_pred)
            - (df["false_positives"] * cost_of_intervention)
        ) / df["Total number of unique outcome events"]

    return df[["benefit_harm", "positive_rate"]]


def plot_cost_benefit_by_ppr(df: pd.DataFrame, per_true_positive: bool) -> pn.ggplot:
    legend_order = sorted(
        df["savings_recources_ratio"].unique(), key=lambda s: int(s.split(":")[0]), reverse=True
    )

    df = df.assign(values=pd.Categorical(df["savings_recources_ratio"], legend_order))

    p = (
        pn.ggplot(df, pn.aes(x="positive_rate", y="benefit_harm", color="savings_recources_ratio"))
        + pn.geom_point()
        + pn.labs(x="Predicted Positive Rate", y="Benefit/Harm")
        + FA_PN_THEME
        + pn.theme(axis_text_x=pn.element_text(rotation=45, hjust=1))
        + pn.labs(color="Profit/Cost ratio")
        + pn.theme(
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            legend_position=(0.3, 0.18),
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
        p += pn.ggtitle("Cost/benefit estimate based on True Positives")
    else:
        p += pn.ggtitle("Cost/benefit estimate based on unique outcomes predicted ≥1")

    for value in legend_order:
        p += pn.geom_path(df[df["savings_recources_ratio"] == value], group=1) # type: ignore
    return p


def cost_benefit_by_savings_recources_ratio(
    run: ForcedAdmissionInpatientPipelineRun,
    per_true_positive: bool,
    savings_from_prevented_outcome: Sequence[int],
    cost_of_intervention: int,
    efficiency_of_intervention: float,
    positive_rates: Sequence[float] = [0.5, 0.2, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01],
) -> pn.ggplot:
    dfs = []

    for savings in savings_from_prevented_outcome:
        df = calculate_cost_benefit(
            run=run,
            savings_from_prevented_outcome=savings,
            cost_of_intervention=cost_of_intervention,
            efficiency_of_intervention=efficiency_of_intervention,
            per_true_positive=per_true_positive,
            positive_rates=positive_rates,
        )

        savings_recources_ratio = (
            f"{int(savings*efficiency_of_intervention)!s}:{cost_of_intervention!s}"
        )
        # Convert to string to allow distinct scales for color
        df["savings_recources_ratio"] = str(savings_recources_ratio)
        df["cost_of_intervention"] = str(cost_of_intervention)
        df["efficiency_of_intervention"] = str(efficiency_of_intervention)
        dfs.append(df)

    plot_df = pd.concat(dfs)

    p = plot_cost_benefit_by_ppr(plot_df, per_true_positive)

    return p


def fa_cost_benefit_by_savings_recources_ratio_and_ppr(
    run: ForcedAdmissionInpatientPipelineRun,
    per_true_positive: bool,
    savings_from_prevented_outcome: Sequence[int],
    cost_of_intervention: int,
    efficiency_of_intervention: float,
    positive_rates: Sequence[float] = [0.5, 0.2, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01],
) -> pn.ggplot:
    p = cost_benefit_by_savings_recources_ratio(
        run=run,
        savings_from_prevented_outcome=savings_from_prevented_outcome,
        cost_of_intervention=cost_of_intervention,
        efficiency_of_intervention=efficiency_of_intervention,
        per_true_positive=per_true_positive,
        positive_rates=positive_rates,
    )

    if per_true_positive:
        p.save(
            filename=run.paper_outputs.paths.figures
            / "fa_inpatient_cost_benefit_estimates_per_true_positives.png",
            width=7,
            height=7,
        )
    else:
        p.save(
            filename=run.paper_outputs.paths.figures
            / "fa_inpatient_cost_benefit_estimates_per_true_unique_outcomes.png",
            width=7,
            height=7,
        )

    return p


if __name__ == "__main__":
    from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    fa_cost_benefit_by_savings_recources_ratio_and_ppr(
        run=get_best_eval_pipeline(),
        per_true_positive=False,
        savings_from_prevented_outcome=[40, 20, 10, 6],
        cost_of_intervention=1,
        efficiency_of_intervention=0.5,
    )

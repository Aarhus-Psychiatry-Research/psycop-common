from collections.abc import Sequence

import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.forced_admission_inpatient.model_eval.config import FA_PN_THEME
from psycop.projects.forced_admission_inpatient.model_eval.model_description.performance.performance_by_ppr import (
    _get_num_of_unique_outcome_events,  # type: ignore
)
from psycop.projects.forced_admission_inpatient.model_eval.model_description.performance.performance_by_ppr import (
    _get_number_of_outcome_events_with_at_least_one_true_positve,  # type: ignore
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def calculate_cost_benefit(run: ForcedAdmissionInpatientPipelineRun, 
                           savings_from_prevented_outcome: int,
                           cost_of_intervention: int,
                           efficiency_of_intervention: float,
                           per_true_positive: bool = True,
                           positive_rates: Sequence[float] = [0.5, 0.2, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01], 
):
    
    output_path = (
        run.paper_outputs.paths.tables
        / 'cost_benefit_estimates.png'
    )
    eval_dataset = run.pipeline_outputs.get_eval_dataset()

    df: pd.DataFrame = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=eval_dataset,
        positive_rates=positive_rates,
    )

    df["Total number of unique outcome events"] = _get_num_of_unique_outcome_events(
        eval_dataset=eval_dataset,
    )

    df["Number of positive outcomes in test set (TP+FN)"] = (
        df["true_positives"] + df["false_negatives"]
    )
    
    df["Number of unique outcome events detected ≥1"] = [
        _get_number_of_outcome_events_with_at_least_one_true_positve(
            eval_dataset=eval_dataset,
            positive_rate=pos_rate,
        )
        for pos_rate in positive_rates
    ]

    saving_per_positive_pred = savings_from_prevented_outcome * efficiency_of_intervention

    df["benefit_harm"] = ((df["true_positives"] * saving_per_positive_pred) - (df["false_positives"]*cost_of_intervention)) / df["Number of positive outcomes in test set (TP+FN)"] 

    df["benefit_harm"] = ((
        df["Number of unique outcome events detected ≥1"] * saving_per_positive_pred
    ) - (df["false_positives"]*cost_of_intervention)) / df["Total number of unique outcome events"]

    return df[['benefit_harm', 'positive_rate']]


def plot_cost_benefit_by_ppr(df: pd.DataFrame,
                              title: str,
                              per_true_positive: bool,
                              savings_from_prevented_outcome: int,
                              cost_of_intervention: int,
                              efficiency_of_intervention: float,

    ) -> pn.ggplot:


    p = (
        pn.ggplot(
            df,
            pn.aes(
                x="cost_benefit",
                y="sensitivity",
                ymin="ci_lower",
                ymax="ci_upper",
                color="actual_positive_rate",
            ),
        )
        + pn.scale_x_discrete()
        + pn.geom_point()
        + pn.geom_linerange(size=0.5)
        + pn.labs(x="Days to outcome", y="Sensitivity")
        + FA_PN_THEME
        + pn.theme(axis_text_x=pn.element_text(rotation=45, hjust=1))
        + pn.scale_color_brewer(type="qual", palette=2)
        + pn.labs(color="Predicted Positive Rate")
        + pn.theme(
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            legend_position=(0.3, 0.88),
        )
    )

    for value in df["actual_positive_rate"].unique():
        p += pn.geom_path(df[df["actual_positive_rate"] == value], group=1)

    return p


if __name__ == "__main__":
    from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    calculate_cost_benefit(run=get_best_eval_pipeline(), savings_from_prevented_outcome=20, cost_of_intervention=1, efficiency_of_intervention=0.5)
    
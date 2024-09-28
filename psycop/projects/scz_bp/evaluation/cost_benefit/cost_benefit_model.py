## Parametrize


# as input:
## EvalDataset
## the distributions
## pprs
## per true positive
## min alert days


from collections.abc import Sequence
from dataclasses import dataclass
from typing import NewType

import numpy as np
import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.forced_admission_outpatient.model_eval.model_description.performance.performance_by_ppr import (
    _get_num_of_unique_outcome_events,
    _get_number_of_outcome_events_with_at_least_one_true_positve,
)

CostBenefitDF = NewType("CostBenefitDF", pl.DataFrame)
CostBenefitDistributionsDF = NewType("CostBenefitDistributionsDF", pl.DataFrame)


@dataclass
class CostBenefitDistributions:
    cost_of_intervention: np.ndarray  # type: ignore
    efficiency_of_intervention: np.ndarray  # type: ignore
    savings_from_prevented_outcome: np.ndarray  # type: ignore

    def get_dataframe(self) -> CostBenefitDistributionsDF:
        return CostBenefitDistributionsDF(
            pl.DataFrame(
                {
                    "cost_of_intervention": self.cost_of_intervention,
                    "efficiency_of_intervention": self.efficiency_of_intervention,
                    "savings_from_prevented_outcome": self.savings_from_prevented_outcome,
                    "cost_benefit_ratio": self.savings_from_prevented_outcome
                    * self.efficiency_of_intervention
                    / self.cost_of_intervention,
                }
            )
        )

    def get_cost_benefit_descriptive_stats(self) -> dict[str, int]:
        # Calculate the required statistics
        df = self.get_dataframe()
        median_value = int(df["cost_benefit_ratio"].median())  # type: ignore
        mean_value = int(df["cost_benefit_ratio"].mean())  # type: ignore
        percentile_5th = int(np.percentile(df["cost_benefit_ratio"], 5))
        percentile_95th = int(np.percentile(df["cost_benefit_ratio"], 95))

        # Add the statistics to a dictionary
        return {
            f"Median ({median_value}:1)": median_value,
            f"Mean ({mean_value}:1)": mean_value,
            f"5th percentile ({percentile_5th}:1)": percentile_5th,
            f"95th percentile ({percentile_95th}:1)": percentile_95th,
        }


@shared_cache().cache()
def cost_benefit_model(
    eval_df: EvalDataset,
    cost_benefit_distributions: CostBenefitDistributions,
    pprs: Sequence[float],
    per_true_positive: bool,
    min_alert_days: int | None,
) -> CostBenefitDF:
    cost_benefit_descriptive_stats = cost_benefit_distributions.get_cost_benefit_descriptive_stats()

    cost_benefit_dfs = []
    for stat, cost_benefit_ratio in cost_benefit_descriptive_stats.items():
        cost_benefit_df = calculate_cost_benefit_from_eval_dataset(
            eval_df=eval_df,
            cost_benefit_ratio=cost_benefit_ratio,
            per_true_positive=per_true_positive,
            min_alert_days=min_alert_days,
            positive_rates=pprs,
        )
        # Convert to string to allow distinct scales for color
        cost_benefit_df["cost_benefit_ratio_str"] = str(stat)
        cost_benefit_dfs.append(cost_benefit_df)
    df = pl.concat(cost_benefit_dfs)
    return CostBenefitDF(df)


def calculate_cost_benefit_from_eval_dataset(
    eval_df: EvalDataset,
    cost_benefit_ratio: int,
    per_true_positive: bool,
    min_alert_days: None | int,
    positive_rates: Sequence[float],
) -> pl.DataFrame:
    df = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=eval_df, positive_rates=positive_rates
    )

    df["Total number of unique outcome events"] = _get_num_of_unique_outcome_events(
        eval_dataset=eval_df
    )

    df["Number of positive outcomes in test set (TP+FN)"] = (
        df["true_positives"] + df["false_negatives"]
    )

    df["Number of unique outcome events detected ≥1"] = [
        _get_number_of_outcome_events_with_at_least_one_true_positve(
            eval_dataset=eval_df, positive_rate=pos_rate, min_alert_days=min_alert_days
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

    return pl.from_pandas(df[["benefit_harm", "positive_rate"]])

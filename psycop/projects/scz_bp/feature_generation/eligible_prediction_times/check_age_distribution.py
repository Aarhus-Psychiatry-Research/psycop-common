import plotnine as pn
import polars as pl

from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.single_filters import (
    SczBpAddAge,
)

if __name__ == "__main__":
    pred_times = SczBpCohort.get_filtered_prediction_times_bundle().prediction_times

    outcome_timestamps = SczBpCohort.get_outcome_timestamps().frame.lazy()
    outcome_with_age = SczBpAddAge().apply(outcome_timestamps)

    first_eligible_outcome = (
        pred_times.frame.join(
            outcome_with_age.collect(), how="left", on="dw_ek_borger", suffix="_outcome"
        )
        .filter(pl.col("age").is_not_null())
        .sort("timestamp")
        .groupby("dw_ek_borger")
        .first()
    )

    (
        pn.ggplot(first_eligible_outcome, pn.aes(x="age"))
        + pn.geom_histogram()
        + pn.labs(x="Age at diagnosis", y="Count")
        + pn.geom_vline(pn.aes(xintercept=60), linetype="dashed")
        + pn.geom_vline(pn.aes(xintercept=15), linetype="dashed")
        + pn.theme_minimal()
    ).save(SCZ_BP_EVAL_OUTPUT_DIR / "age_dist_plot.png", dpi=300)

    (
        pn.ggplot(first_eligible_outcome, pn.aes(x="age"))
        + pn.stat_ecdf()
        + pn.labs(x="Age at diagnosis", y="Cumulative proportion")
        + pn.geom_vline(pn.aes(xintercept=60), linetype="dashed")
        + pn.geom_vline(pn.aes(xintercept=15), linetype="dashed")
        + pn.theme_minimal()
    ).save(SCZ_BP_EVAL_OUTPUT_DIR / "age_dist_cum.png", dpi=300)

    for max_age in [40, 50, 60]:
        filtered_by_age = first_eligible_outcome.filter(pl.col("age") < max_age)
        print(f"Max age: {max_age}")
        print(f"\tN positive cases: {filtered_by_age.shape[0]}")
    print("No max age")
    print(f"\tN positive cases: {first_eligible_outcome.shape[0]}")

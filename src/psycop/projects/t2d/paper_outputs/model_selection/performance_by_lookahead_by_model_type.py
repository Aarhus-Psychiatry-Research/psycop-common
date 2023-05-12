import polars as pl
from psycop.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    days_from_first_positive_to_diagnosis,
)
from psycop.projects.t2d.paper_outputs.config import (
    BEST_POS_RATE,
    DEVELOPMENT_GROUP,
    EVAL_RUN,
)
from psycop.projects.t2d.utils.best_runs import Run, RunGroup


def get_performance_for_group(group: RunGroup, pos_rate: float) -> pl.DataFrame:
    best_runs_by_lookahead = group.get_best_runs_by_lookahead()

    performance_dfs: list[pl.DataFrame] = []

    for run_name in best_runs_by_lookahead["run_name"]:
        run_object = Run(group=group, name=run_name, pos_rate=pos_rate)

        performance_dfs.append(get_performance_for_run(run=run_object))

    return pl.concat(performance_dfs, how="vertical")


def get_performance_for_run(run: Run) -> pl.DataFrame:
    return_dict = {
        "model_name": run.cfg.model.name,
        "lookahead_days": float(run.cfg.preprocessing.pre_split.min_lookahead_days),
        "run_name": run.name,
        "auroc": run.get_auroc(),
        "mean_warning_days": get_mean_days_from_first_positive_to_diagnosis_for_run(
            run=run
        ),
    }

    return pl.DataFrame(return_dict)


def get_mean_days_from_first_positive_to_diagnosis_for_run(run: Run) -> float:
    return days_from_first_positive_to_diagnosis(
        eval_dataset=run.get_eval_dataset(),
        positive_rate=run.pos_rate,
        aggregation_method="mean",
    )


def get_publication_ready_performance_for_group(run_group: RunGroup):
    df = get_performance_for_group(group=run_group, pos_rate=BEST_POS_RATE)

    auroc_df = df.pivot(
        index=["model_name"],
        columns=["lookahead_days"],
        values=["auroc"],
        aggregate_function="first",
    ).select(
        pl.col("model_name"),
        pl.lit("AUROC").alias("measure"),
        pl.all().exclude("model_name").round(2),
    )

    warning_days_df = df.pivot(
        index=["model_name"],
        columns=["lookahead_days"],
        values=["mean_warning_days"],
        aggregate_function="first",
    ).select(
        pl.col("model_name"),
        pl.lit(
            f"Mean years from first positive to event at {BEST_POS_RATE} positive rate"
        ).alias("measure"),
        (pl.all().exclude("model_name") / 365.25).round(1),
    )

    df = pl.concat([auroc_df, warning_days_df]).sort(["model_name"], descending=True)

    return df


if __name__ == "__main__":
    run_group = DEVELOPMENT_GROUP

    df = get_publication_ready_performance_for_group(run_group=run_group)

    pass

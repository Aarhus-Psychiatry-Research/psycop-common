import polars as pl

from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    days_from_first_positive_to_diagnosis,
)
from psycop.projects.scz_bp.evaluation.configs import (
    BEST_POS_RATE,
    DEVELOPMENT_GROUP,
    SCZ_BP_CUSTOM_COLUMNS,
)
from psycop.projects.scz_bp.evaluation.pipeline_objects import PipelineRun, RunGroup


def get_performance_for_group(run_group: RunGroup, pos_rate: float) -> pl.DataFrame:
    best_runs_by_lookahead = run_group.get_best_runs_by_lookahead()

    performance_dfs: list[pl.DataFrame] = []

    for run_name in best_runs_by_lookahead["run_name"]:
        run_object = PipelineRun(group=run_group, name=run_name, pos_rate=pos_rate)

        performance_dfs.append(get_performance_for_run(run=run_object))

    return pl.concat(performance_dfs, how="vertical")


def get_performance_for_run(run: PipelineRun) -> pl.DataFrame:
    return_dict = {
        "model_name": run.inputs.cfg.model.name,
        "lookahead_days": float(
            run.inputs.cfg.preprocessing.pre_split.min_lookahead_days,
        ),
        "run_name": run.name,
        "group_name": run.group.name,
        "auroc": run.pipeline_outputs.get_auroc(),
        "median": get_median_days_from_first_positive_to_diagnosis_for_run(
            run=run,
        ),
    }

    return pl.DataFrame(return_dict)


def get_median_days_from_first_positive_to_diagnosis_for_run(run: PipelineRun) -> float:
    return days_from_first_positive_to_diagnosis(
        eval_dataset=run.pipeline_outputs.get_eval_dataset(
            custom_columns=SCZ_BP_CUSTOM_COLUMNS,
        ),
        positive_rate=run.paper_outputs.pos_rate,
        aggregation_method="median",
    )


def get_publication_ready_performance_for_group(run_group: RunGroup) -> pl.DataFrame:
    df = get_performance_for_group(run_group=run_group, pos_rate=BEST_POS_RATE)

    auroc_df = df.pivot(
        index=["group_name", "model_name"],
        columns=["lookahead_days"],
        values=["auroc"],
        aggregate_function="first",
    ).select(
        pl.col("model_name"),
        pl.col("group_name"),
        pl.lit("AUROC").alias("measure"),
        pl.all().exclude("group_name", "model_name").round(2),
    )

    warning_days_df = df.pivot(
        index=["group_name", "model_name"],
        columns=["lookahead_days"],
        values=["median"],
        aggregate_function="first",
    ).select(
        pl.col("model_name"),
        pl.col("group_name"),
        pl.lit(
            f"Median years from first positive to event at {BEST_POS_RATE} predicted positive rate",
        ).alias("measure"),
        (pl.all().exclude("group_name", "model_name") / 365.25).round(1),
    )

    df = pl.concat([auroc_df, warning_days_df]).sort(["model_name"], descending=True)

    return df


run_group = DEVELOPMENT_GROUP

best_runs_in_group = get_performance_for_group(DEVELOPMENT_GROUP, pos_rate=0.03)
run_with_max_auc = (
    best_runs_in_group.filter(pl.col("auroc") == pl.col("auroc").max())
    .select("run_name")
    .item()
)

DEVELOPMENT_PIPELINE_RUN = PipelineRun(
    group=DEVELOPMENT_GROUP,
    name=run_with_max_auc,
    pos_rate=0.03,
)

if __name__ == "__main__":
    df = get_publication_ready_performance_for_group(run_group=run_group)

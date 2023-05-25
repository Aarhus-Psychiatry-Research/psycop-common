import polars as pl
from psycop.projects.t2d.paper_outputs.aggregate_eval.md_objects import (
    create_supplementary_from_markdown_artifacts,
)
from psycop.projects.t2d.paper_outputs.aggregate_eval.single_pipeline_full_eval import (
    t2d_main_manuscript_eval,
)
from psycop.projects.t2d.paper_outputs.config import BEST_POS_RATE, DEVELOPMENT_GROUP
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun, RunGroup
from wasabi import Printer

msg = Printer(timestamp=True)


def get_best_runs_from_model_type(
    dev_run_group: RunGroup,
    model_type: str,
) -> tuple[PipelineRun]:
    """Get PipelineRun objects for the best runs of a given model type"""
    performance_df = pl.from_pandas(dev_run_group.all_runs_performance_df)

    best_run_df = (
        performance_df.filter(pl.col("model_name") == model_type)
        .sort(by="roc_auc", descending=True)
        .groupby(["model_name", "lookahead_days"])
        .head(1)
    )

    best_run_names = list(best_run_df["run_name"])
    best_runs = tuple(
        PipelineRun(
            group=dev_run_group,
            name=name,
            pos_rate=BEST_POS_RATE,
        )
        for name in best_run_names
    )

    msg.info(f"Evaluating {[run.name for run in best_runs]}")

    return best_runs


def full_eval_for_supplementary(dev_run_group: RunGroup) -> None:
    """Run full evaluation for the supplementary material."""
    best_model_type = "xgboost"
    best_runs_from_model_type = get_best_runs_from_model_type(
        dev_run_group=dev_run_group,
        model_type=best_model_type,
    )

    artifacts = []

    for run in best_runs_from_model_type:
        artifacts += t2d_main_manuscript_eval(dev_pipeline=run)

    create_supplementary_from_markdown_artifacts(
        artifacts=artifacts, first_table_index=4, first_figure_index=3
    )


if __name__ == "__main__":
    full_eval_for_supplementary(dev_run_group=DEVELOPMENT_GROUP)

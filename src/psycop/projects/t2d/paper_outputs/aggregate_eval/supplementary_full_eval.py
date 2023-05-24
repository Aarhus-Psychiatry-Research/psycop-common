import polars as pl
from psycop.projects.t2d.paper_outputs.aggregate_eval.single_pipeline_full_eval import (
    t2d_main_manuscript_eval,
)
from psycop.projects.t2d.paper_outputs.config import DEVELOPMENT_GROUP
from psycop.projects.t2d.utils.best_runs import PipelineRun, RunGroup


def get_best_runs_from_model_type(
    dev_run_group: RunGroup, model_type: str
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
        PipelineRun(group=dev_run_group, name=name, pos_rate=0.03)
        for name in best_run_names
    )

    return best_runs


def full_eval_for_supplementary(dev_run_group: RunGroup) -> None:
    """Run full evaluation for the supplementary material."""
    best_model_type = "xgboost"
    best_runs_from_model_type = get_best_runs_from_model_type(
        dev_run_group=dev_run_group, model_type=best_model_type
    )

    for run in best_runs_from_model_type:
        t2d_main_manuscript_eval(dev_pipeline=run)


if __name__ == "__main__":
    full_eval_for_supplementary(dev_run_group=DEVELOPMENT_GROUP)

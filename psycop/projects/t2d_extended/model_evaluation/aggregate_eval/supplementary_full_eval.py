import polars as pl
from wasabi import Printer

from psycop.common.model_evaluation.markdown.md_objects import (
    MarkdownArtifact,
    create_supplementary_from_markdown_artifacts,
)
from psycop.projects.t2d_extended.model_evaluation.aggregate_eval.single_pipeline_full_eval import (
    t2d_extended_create_markdown_artifacts,
    t2d_extended_main_manuscript_eval,
)
from psycop.projects.t2d_extended.model_evaluation.config import BEST_POS_RATE, DEVELOPMENT_GROUP
from psycop.projects.t2d_extended.model_evaluation.run_pipeline_on_val import test_pipeline
from psycop.projects.t2d_extended.model_evaluation.selected_runs import get_best_eval_pipeline
from psycop.projects.t2d_extended.utils.pipeline_objects import EVAL_ROOT, RunGroup, T2DExtendedPipelineRun

msg = Printer(timestamp=True)


def get_best_runs_from_model_type(
    dev_run_group: RunGroup, model_type: str
) -> tuple[T2DExtendedPipelineRun]:
    """Get PipelineRun objects for the best runs of a given model type"""
    performance_df = pl.from_pandas(dev_run_group.all_runs_performance_df)

    best_run_df = (
        performance_df.filter(pl.col("model_name") == model_type)
        .sort(by=["roc_auc"], descending=True)
        .groupby(["model_name", "lookahead_days"])
        .head(1)
        .sort(by=["lookahead_days"], descending=False)
    )

    best_run_names = list(best_run_df["run_name"])

    best_runs = tuple(
        T2DExtendedPipelineRun(
            group=dev_run_group,
            name=name,
            pos_rate=BEST_POS_RATE,
            create_output_paths_on_init=False,
        )
        for name in best_run_names
    )

    msg.divider(f"Evaluating {[run.name for run in best_runs]}")

    return best_runs  # type: ignore


def get_test_artifacts_for_pipeline(
    recreate_artifacts: bool, dev_pipeline: T2DExtendedPipelineRun
) -> list[MarkdownArtifact]:
    """Get the standard set of artifacts for a given pipeline"""
    pipeline = test_pipeline(pipeline_to_test=dev_pipeline)

    if recreate_artifacts:
        current_run_artifacts = t2d_extended_main_manuscript_eval(pipeline=pipeline)
    else:
        msg.warn(f"recreate_artifacts set to {recreate_artifacts}: Not recreating artifacts")
        current_run_artifacts = t2d_extended_create_markdown_artifacts(pipeline=pipeline)

    run_md = create_supplementary_from_markdown_artifacts(
        artifacts=current_run_artifacts,
        table_title_prefix="Supplementary Table",
        figure_title_prefix="Supplementary Figure",
    )

    with (
        EVAL_ROOT
        / f"supplementary_{pipeline.inputs.cfg.preprocessing.pre_split.min_lookahead_days}_{pipeline.model_type}_{pipeline.name}.md"
    ).open("w") as f:
        f.write(run_md)

    return current_run_artifacts


def full_eval_for_supplementary(dev_run_group: RunGroup, recreate_artifacts: bool = True) -> None:
    """Run full evaluation for the supplementary material."""
    best_model_type = "xgboost"

    best_runs_from_model_type = get_best_runs_from_model_type(
        dev_run_group=dev_run_group, model_type=best_model_type
    )

    artifacts = []

    for run in best_runs_from_model_type:
        run_artifacts = get_test_artifacts_for_pipeline(
            recreate_artifacts=recreate_artifacts, dev_pipeline=run
        )

        # Do not add the main pipeline's eval to supplementary
        if run.name != get_best_eval_pipeline().name:
            artifacts += run_artifacts

    combined_supplementary_md = create_supplementary_from_markdown_artifacts(
        artifacts=artifacts, first_table_index=4, first_figure_index=3
    )

    with (EVAL_ROOT / f"supplementary_combined_{dev_run_group.name}.md").open("w") as f:
        f.write(combined_supplementary_md)


if __name__ == "__main__":
    full_eval_for_supplementary(dev_run_group=DEVELOPMENT_GROUP, recreate_artifacts=True)

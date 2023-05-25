from psycop.projects.t2d.paper_outputs.config import BEST_EVAL_PIPELINE, DEV_GROUP_NAME
from psycop.projects.t2d.paper_outputs.model_description.performance.main_performance_figure import (
    t2d_create_main_performance_figure,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.performance_by_ppr import (
    t2d_output_performance_by_ppr,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.main_robustness_figure import (
    t2d_create_main_robustness_figure,
)
from psycop.projects.t2d.paper_outputs.run_pipeline_on_train import (
    get_test_pipeline_run,
)
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun, RunGroup
from wasabi import Printer

msg = Printer(timestamp=True)


def t2d_md_main_performance_figure(run: PipelineRun) -> str:
    return f"""### Performance of {run.model_type.capitalize()} with {int(run.inputs.cfg.preprocessing.pre_split.min_lookahead_days / 365)} years of lookahead
![Performance]({run.paper_outputs.artifact_path / run.paper_outputs.artifact_names.main_performance_figure})

"""


def t2d_concat_results_to_md_objects(run: PipelineRun) -> None:
    md = ""

    md += t2d_md_main_performance_figure(run=run)
    md += t2d_md_main_robustness_figure(run=run)
    md += t2d_md_main_performance_by_ppr_figure(run=run)


def t2d_main_manuscript_eval(dev_pipeline: PipelineRun) -> None:
    train_pipeline = get_test_pipeline_run(pipeline_to_train=dev_pipeline)

    msg.info(
        f"Generating main manuscript eval to {train_pipeline.paper_outputs.artifact_path}"
    )

    t2d_create_main_performance_figure(run=train_pipeline)
    t2d_create_main_robustness_figure(run=train_pipeline)
    t2d_output_performance_by_ppr(run=train_pipeline)

    t2d_concat_results_to_md_objects(run=train_pipeline)


if __name__ == "__main__":
    t2d_main_manuscript_eval(
        dev_pipeline=BEST_EVAL_PIPELINE,
    )

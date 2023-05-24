from psycop.projects.t2d.paper_outputs.config import BEST_DEV_RUN
from psycop.projects.t2d.paper_outputs.create_patchwork_figure import (
    t2d_create_patchwork_figure,
)
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
    run_pipeline_on_train,
)
from psycop.projects.t2d.utils.best_runs import PipelineRun


def t2d_main_manuscript_eval(dev_pipeline: PipelineRun) -> None:
    train_pipeline = run_pipeline_on_train(pipeline_to_train=dev_pipeline)
    t2d_create_main_performance_figure(run=train_pipeline)
    t2d_create_main_robustness_figure(run=train_pipeline)
    t2d_output_performance_by_ppr(run=train_pipeline)


if __name__ == "__main__":
    t2d_main_manuscript_eval(dev_pipeline=BEST_DEV_RUN)

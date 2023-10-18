from psycop.projects.t2d.paper_outputs.config import (
    BEST_POS_RATE,
    DEVELOPMENT_GROUP,
)
from psycop.projects.t2d.paper_outputs.run_pipeline_on_train import (
    test_pipeline,
)
from psycop.projects.t2d.utils.pipeline_objects import T2DPipelineRun


def get_best_dev_pipeline() -> T2DPipelineRun:
    return T2DPipelineRun(
        group=DEVELOPMENT_GROUP,
        name="nonviolentstigmaria",
        pos_rate=BEST_POS_RATE,
        additional_cfg_keys={
            "project": {
                "project_path": "E:/shared_resources/t2d",
                "seed": "42",
                "gpu": "true",
                "name": "nonviolentstigmaria",
            },
        },
        remove_cfg_keys={"name", "project_path", "seed", "gpu", "wandb"},
    )


def get_best_eval_pipeline() -> T2DPipelineRun:
    return test_pipeline(pipeline_to_test=get_best_dev_pipeline())

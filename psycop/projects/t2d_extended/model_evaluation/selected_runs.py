from psycop.projects.t2d_extended.model_evaluation.config import BEST_POS_RATE, DEVELOPMENT_GROUP
from psycop.projects.t2d_extended.model_evaluation.run_pipeline_on_val import test_pipeline
from psycop.projects.t2d_extended.utils.pipeline_objects import T2DExtendedPipelineRun


def get_best_dev_pipeline() -> T2DExtendedPipelineRun:
    return T2DExtendedPipelineRun(
        group=DEVELOPMENT_GROUP,
        name="2018-01-01_2022-01-01_2022-12-31", # TODO fh: change
        pos_rate=BEST_POS_RATE,
        additional_cfg_keys={
            "project": {
                "project_path": "E:/shared_resources/t2d_extended",
                "seed": "42",
                "gpu": "true",
                "name": "2018-01-01_2022-01-01_2022-12-31", # TODO fh: change
            }
        },
        remove_cfg_keys={"name", "project_path", "seed", "gpu", "wandb"}, # TODO fh: check, compare with forced admissions
    )


def get_best_eval_pipeline() -> T2DExtendedPipelineRun:
    return test_pipeline(pipeline_to_test=get_best_dev_pipeline())

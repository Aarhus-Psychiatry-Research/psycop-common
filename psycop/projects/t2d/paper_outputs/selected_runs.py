from psycop.projects.t2d.paper_outputs.config import (
    BEST_POS_RATE,
    DEVELOPMENT_GROUP,
)
from psycop.projects.t2d.paper_outputs.run_pipeline_on_train import (
    test_pipeline,
)
from psycop.projects.t2d.utils.pipeline_objects import T2DPipelineRun

BEST_DEV_PIPELINE = T2DPipelineRun(
    group=DEVELOPMENT_GROUP,
    name="nonviolentstigmaria",
    pos_rate=BEST_POS_RATE,
    additional_cfg_keys={"project": {"project_path": "E:/shared_resources/t2d"}},
)
BEST_EVAL_PIPELINE = test_pipeline(pipeline_to_test=BEST_DEV_PIPELINE)

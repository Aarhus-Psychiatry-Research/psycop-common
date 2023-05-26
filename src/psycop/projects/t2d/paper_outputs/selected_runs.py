from psycop.projects.t2d.paper_outputs.config import (
    BEST_POS_RATE,
    DEVELOPMENT_GROUP,
    EVAL_GROUP,
)
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun

BEST_DEV_PIPELINE = PipelineRun(
    group=DEVELOPMENT_GROUP,
    name="surefootedlygoatpox",
    pos_rate=BEST_POS_RATE,
)
BEST_EVAL_PIPELINE = PipelineRun(
    group=EVAL_GROUP,
    name="pseudoreformatoryhizz",
    pos_rate=BEST_POS_RATE,
)

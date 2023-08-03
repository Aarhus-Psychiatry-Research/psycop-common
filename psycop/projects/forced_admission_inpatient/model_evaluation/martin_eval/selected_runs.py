from psycop.projects.forced_admission_inpatient.model_evaluation.martin_eval.config import (
    BEST_POS_RATE,
    BEST_RUN_NAME,
    DEVELOPMENT_GROUP,
    EVAL_GROUP,
)
from psycop.projects.forced_admission_inpatient.model_evaluation.martin_eval.run_pipeline_on_test import (
    test_pipeline,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    PipelineRun,
)

BEST_DEV_PIPELINE = PipelineRun(
    group=DEVELOPMENT_GROUP,
    name=BEST_RUN_NAME,
    paper_outputs_path=EVAL_GROUP.group_dir / f"{BEST_RUN_NAME}-eval-on-test",
    pos_rate=BEST_POS_RATE,
)
BEST_EVAL_PIPELINE = test_pipeline(pipeline_to_test=BEST_DEV_PIPELINE)

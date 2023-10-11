from psycop.projects.forced_admission_inpatient.model_eval.config import (
    BEST_POS_RATE,
    DEVELOPMENT_GROUP,
)
from psycop.projects.forced_admission_inpatient.model_eval.run_pipeline_on_val import (
    test_pipeline,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    PipelineRun,
)

BEST_DEV_PIPELINE = PipelineRun(
    group=DEVELOPMENT_GROUP,
    name=DEVELOPMENT_GROUP.get_best_runs_by_lookahead()[
        0,
        2,
    ],  # [0,2] for best logistic regression and [1,2] for best xgboost
    pos_rate=BEST_POS_RATE,
    create_output_paths_on_init=False,
)
BEST_EVAL_PIPELINE = test_pipeline(
    pipeline_to_test=BEST_DEV_PIPELINE, splits_for_evaluation=["val_with_washout"]
)

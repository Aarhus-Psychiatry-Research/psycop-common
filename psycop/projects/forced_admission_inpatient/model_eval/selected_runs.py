from psycop.projects.forced_admission_inpatient.model_eval.config import (
    BEST_POS_RATE,
    DEVELOPMENT_GROUP,
)
from psycop.projects.forced_admission_inpatient.model_eval.run_pipeline_on_val import (
    test_selected_model_pipeline,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    PipelineRun,
)

get_best_dev_pipeline = PipelineRun(
    group=DEVELOPMENT_GROUP,
    name=DEVELOPMENT_GROUP.get_best_runs_by_lookahead()[
        0,
        2,
    ],  # [0,2] for best logistic regression and [1,2] for best xgboost
    pos_rate=BEST_POS_RATE,
    create_output_paths_on_init=False,
)

get_best_eval_pipeline = test_selected_model_pipeline(
    pipeline_to_test=get_best_dev_pipeline,
    datasets_for_evaluation=["val_with_washout"],
)

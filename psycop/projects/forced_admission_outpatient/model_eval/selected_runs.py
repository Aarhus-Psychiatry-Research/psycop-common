from psycop.projects.forced_admission_outpatient.model_eval.config import (
    BEST_POS_RATE,
    DEVELOPMENT_GROUP,
    MODEL_ALGORITHM,
)
from psycop.projects.forced_admission_outpatient.model_eval.run_pipeline_on_val import (
    test_selected_model_pipeline,
)
from psycop.projects.forced_admission_outpatient.utils.pipeline_objects import (
    ForcedAdmissionOutpatientPipelineRun,
)


def get_best_dev_pipeline() -> ForcedAdmissionOutpatientPipelineRun:
    return ForcedAdmissionOutpatientPipelineRun(
        group=DEVELOPMENT_GROUP,
        name=DEVELOPMENT_GROUP.get_best_runs_by_lookahead()[MODEL_ALGORITHM, 2],
        pos_rate=BEST_POS_RATE,
        create_output_paths_on_init=False,
    )


def get_best_eval_pipeline() -> ForcedAdmissionOutpatientPipelineRun:
    return test_selected_model_pipeline(
        pipeline_to_test=get_best_dev_pipeline(),
        splits_for_training=["train"],
        splits_for_evaluation=["val"],  # add with_washout if eval on cohort with washout
    )
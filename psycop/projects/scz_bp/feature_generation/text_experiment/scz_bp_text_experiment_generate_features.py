from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.feature_generation.text_experiment.scz_bp_text_experiment_feature_spec import (
    SczBpTextExperimentFeatures,
)

if __name__ == "__main__":
    generate_feature_set(
        project_info=ProjectInfo(
            project_name="scz_bp", project_path=OVARTACI_SHARED_DIR / "scz_bp" / "text_exp"
        ),
        eligible_prediction_times=SczBpCohort.get_filtered_prediction_times_bundle().prediction_times.frame.to_pandas(),
        feature_specs=SczBpTextExperimentFeatures().get_feature_specs(lookbehind_days=[730]),
        # generate_in_chunks=True,  # noqa: ERA001
        # chunksize=10,  # noqa: ERA001
        feature_set_name="text_exp_730d",
    )

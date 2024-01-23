"""Main feature generation."""
import sys
from pathlib import Path

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    init_wandb_and_generate_feature_set,
)
from psycop.projects.cancer.cancer_config import (
    get_cancer_feature_specifications,
    get_cancer_project_info,
)
from psycop.projects.cancer.feature_generation.cohort_definition.cancer_cohort_definer import (
    CancerCohortDefiner,
)

if __name__ == "__main__":
    # will not run without this chunk, should be fixed properly at some point
    if sys.platform == "win32":
        (Path(__file__).resolve().parents[0] / "wandb" / "debug-cli.onerm").mkdir(
            exist_ok=True, parents=True
        )

    init_wandb_and_generate_feature_set(
        project_info=get_cancer_project_info(),
        eligible_prediction_times=CancerCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.frame.to_pandas(),
        feature_specs=get_cancer_feature_specifications(),
        generate_in_chunks=True,
        chunksize=10,
        feature_set_name="with_sentence_transformer",
    )

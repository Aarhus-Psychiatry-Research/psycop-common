"""Main feature generation."""

import datetime as dt
import logging

import polars as pl

from psycop.common.cohort_definition import PredictionTimeFrame
from psycop.common.feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset,
)
from psycop.projects.restraint.cohort.restraint_cohort_definer import RestraintCohortDefiner
from psycop.projects.restraint.feature_generation.modules.specify_text_features import (
    TextFeatureSpecifier,
)
from psycop.projects.restraint.restraint_global_config import RESTRAINT_PROJECT_INFO

log = logging.getLogger()


def main(note_type: str, model_name: str):
    """Main function for loading, generating and evaluating a flattened
    dataset."""
    project_info = RESTRAINT_PROJECT_INFO

    restraint_pred_times = (
        RestraintCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.to_pandas()  # type: ignore
    )

    feature_specs = TextFeatureSpecifier(
        project_info=project_info, min_set_for_debug=False
    ).get_text_feature_specs(note_type=note_type, model_name=model_name)

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,  # type: ignore
        prediction_times_frame=PredictionTimeFrame(pl.DataFrame(restraint_pred_times)),
        n_workers=None,
        step_size=dt.timedelta(days=200),
    )

    flattened_df.write_parquet(project_info.project_path / f"{note_type}_{model_name}.parquet")


if __name__ == "__main__":
    project_info = RESTRAINT_PROJECT_INFO

    main("all_relevant", "tfidf-1000")
    main("all_relevant", "dfm-encoder-large")
    main("all_relevant", "dfm-encoder-large-v1-finetuned")

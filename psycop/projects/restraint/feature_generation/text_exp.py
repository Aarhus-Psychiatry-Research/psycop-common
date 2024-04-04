"""Main feature generation."""

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


def main():
    """Main function for loading, generating and evaluating a flattened
    dataset."""
    project_info = RESTRAINT_PROJECT_INFO

    restraint_pred_times = (
        RestraintCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.to_pandas()
    )
    chunk1 = restraint_pred_times[: int(len(restraint_pred_times) / 3)]
    chunk2 = restraint_pred_times[
        int(len(restraint_pred_times) / 3) : int(len(restraint_pred_times) / 3) * 2
    ]
    chunk3 = restraint_pred_times[int(len(restraint_pred_times) / 3) * 2 :]

    for count, chunk in enumerate([chunk1, chunk2, chunk3]):
        for note_type in ["aktuelt_psykisk", "all_relevant"]:
            for model_name in [
                "tfidf-500",
                "tfidf-1000",
                "dfm-encoder-large",
                "dfm-encoder-large-v1-finetuned",
            ]:
                feature_specs = TextFeatureSpecifier(
                    project_info=project_info, min_set_for_debug=False
                ).get_text_feature_specs(note_type=note_type, model_name=model_name)

                flattened_df = create_flattened_dataset(
                    feature_specs=feature_specs,  # type: ignore
                    prediction_times_frame=PredictionTimeFrame(pl.DataFrame(chunk)),
                    n_workers=None,
                    compute_lazily=False,
                )

                flattened_df.write_parquet(
                    project_info.project_path / f"{note_type}_{model_name}_{count}.parquet"
                )


if __name__ == "__main__":
    project_info = RESTRAINT_PROJECT_INFO

    main()

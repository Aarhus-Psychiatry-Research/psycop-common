"""Test that the descriptive stats table is generated correctly."""

from psycop_model_training.model_eval.base_artifacts.tables.descriptive_stats_table import (
    DescriptiveStatsTable,
)
from psycop_model_training.model_eval.dataclasses import ArtifactContainer, EvalDataset


def test_generate_descriptive_stats_table(synth_eval_dataset: EvalDataset):
    """Test that table is generated correctly."""

    table1 = DescriptiveStatsTable(
        eval_dataset=synth_eval_dataset,
    )

    table_spec = ArtifactContainer(
        label="table_1",
        artifact=table1.generate_descriptive_stats_table(),
    )

    output_table = table_spec.artifact

    # Assert that there are no NaN values in the table
    assert not output_table.isnull().values.any()

    # Assert that all generic statistics are present
    assert (
        output_table["category"]
        .isin(
            [
                "(visit_level) age (mean / interval)",
                "(visit_level) visits followed by positive outcome",
                "(patient_level) patients_with_positive_outcome",
                "(patient level) time_to_first_positive_outcome",
            ],
        )
        .any()
    )

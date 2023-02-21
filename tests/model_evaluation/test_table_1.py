"""Test that table is generated correctly."""

# pylint: disable=missing-function-docstring
import pandas as pd

from psycop_model_training.model_eval.base_artifacts.tables.tables import (
    generate_table_1,
)
from psycop_model_training.model_eval.dataclasses import ArtifactContainer, EvalDataset


def test_generate_table_1(synth_eval_dataset: EvalDataset):
    """Test that table is generated correctly."""
    table_spec = ArtifactContainer(
        label="table_1",
        artifact=generate_table_1(
            eval_dataset=synth_eval_dataset,
            output_format="df",
        ),
    )

    output_table = table_spec.artifact

    assert isinstance(output_table, pd.DataFrame)
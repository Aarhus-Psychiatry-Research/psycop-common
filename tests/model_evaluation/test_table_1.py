"""Test that table is generated correctly."""
import pandas as pd

from psycop_model_training.model_eval.base_artifacts.tables.table_1 import Table1
from psycop_model_training.model_eval.dataclasses import ArtifactContainer, EvalDataset


def test_generate_table_1(synth_eval_dataset: EvalDataset):
    """Test that table is generated correctly."""

    table1 = Table1(
        eval_dataset=synth_eval_dataset,
    )

    table_spec = ArtifactContainer(
        label="table_1",
        artifact=table1.generate_table_1(),
    )

    output_table = table_spec.artifact

    assert not output_table.isnull().values.any()
    assert output_table.shape == (11, 5)

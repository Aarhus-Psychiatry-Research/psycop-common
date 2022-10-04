"""table_test_auc_by_group_table."""
# pylint: disable=missing-function-docstring

from psycopt2d.tables import auc_by_group_table
from psycopt2d.utils import bin_continuous_data


def test_auc_by_group_table(synth_data):
    synth_data["Age bins"] = bin_continuous_data(
        synth_data["age"],
        bins=[0, 18, 30, 50, 120],
    )

    table = auc_by_group_table(
        synth_data,
        pred_probs_col_name="pred_prob",
        outcome_col_name="label",
        groups=["gender", "Age bins"],
    )
    # Stupid test, mainly to remember we have the function..
    assert table.index.names == ["Group", "Value"]

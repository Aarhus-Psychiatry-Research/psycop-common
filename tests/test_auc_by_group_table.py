from psycopt2d.tables import auc_by_group_table


def test_auc_by_group_table(synth_data):
    table = auc_by_group_table(
        synth_data,
        pred_probs_col_name="pred_prob",
        outcome_col_name="label",
        categorical_groups=["gender"],
        age_col_name="age",
    )
    assert table.index.names == ["Group", "Value"]

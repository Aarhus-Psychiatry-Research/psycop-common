import numpy as np
import pandas as pd
import pytest

from psycop.projects.fairness.bootstrap import cluster_bootstrap


@pytest.fixture
def test_data() -> pd.DataFrame:
    """Create a test dataframe."""
    return pd.DataFrame(
        {
            "dw_ek_borger": [1, 1, 2, 3, 3, 3, 4],
            "timestamp": [101, 102, 201, 301, 302, 303, 401],
            "sensitive_attribute": ["F", "F", "F", "M", "M", "M", "M"],
            "y": [1, 0, 1, 1, 0, 1, 0],
            "y_pred": [1, 0, 1, 0, 0, 1, 0],
        }
    )


def test_bootstrap_keeps_all_encounters(test_data: pd.DataFrame):
    boot = cluster_bootstrap(df=test_data, rng=np.random.default_rng(2))

    original = test_data.groupby("dw_ek_borger")["timestamp"].apply(set)

    boot_patients = boot.groupby("_bootstrap_id")["timestamp"].apply(set)

    for bootstrap_patient, encounters in boot_patients.items():
        patient_id = boot.loc[boot["_bootstrap_id"] == bootstrap_patient, "dw_ek_borger"].iloc[0]

        assert encounters == original.loc[patient_id]


def test_bootstrap_has_same_number_of_patient_draws(test_data: pd.DataFrame):
    boot = cluster_bootstrap(
        test_data, sampling_unit_col="dw_ek_borger", rng=np.random.default_rng(2)
    )

    assert boot["_bootstrap_id"].nunique() == test_data["dw_ek_borger"].nunique()


def test_duplicate_patients_get_unique_bootstrap_ids(test_data: pd.DataFrame):
    boot = cluster_bootstrap(
        test_data, sampling_unit_col="dw_ek_borger", rng=np.random.default_rng(2)
    )

    duplicated_real_patients = boot.groupby("dw_ek_borger")["_bootstrap_id"].nunique()

    assert duplicated_real_patients.max() > 1


def test_stratified_bootstrap_preserves_group_counts(test_data: pd.DataFrame):
    boot = cluster_bootstrap(
        test_data,
        sampling_unit_col="dw_ek_borger",
        stratify_col="sensitive_attribute",
        rng=np.random.default_rng(2),
    )

    sampled_patient_table = boot[["_bootstrap_id", "sensitive_attribute"]].drop_duplicates()

    expected = (
        test_data[["dw_ek_borger", "sensitive_attribute"]]
        .drop_duplicates()["sensitive_attribute"]
        .value_counts()
    )

    observed = sampled_patient_table["sensitive_attribute"].value_counts()

    pd.testing.assert_series_equal(observed, expected)


def test_weights_sum_to_one(test_data: pd.DataFrame):
    boot = cluster_bootstrap(
        test_data,
        sampling_unit_col="dw_ek_borger",
        sample_weight=True,
        rng=np.random.default_rng(2),
    )

    weight_sum = boot.groupby("_bootstrap_id")["sample_weight"].sum()

    assert weight_sum.eq(1).all()


def test_patient_weight_values(test_data: pd.DataFrame):
    boot = cluster_bootstrap(
        test_data,
        sampling_unit_col="dw_ek_borger",
        sample_weight=True,
        rng=np.random.default_rng(2),
    )

    counts = boot.groupby("_bootstrap_id")["dw_ek_borger"].count()

    weights = boot.groupby("_bootstrap_id")["sample_weight"].first()

    expected = 1 / counts

    pd.testing.assert_series_equal(weights.sort_index(), expected.sort_index())


def test_bootstrap_is_reproducible(test_data: pd.DataFrame):
    boot1 = cluster_bootstrap(
        test_data,
        sampling_unit_col="dw_ek_borger",
        sample_weight=True,
        rng=np.random.default_rng(2),
    )

    boot2 = cluster_bootstrap(
        test_data,
        sampling_unit_col="dw_ek_borger",
        sample_weight=True,
        rng=np.random.default_rng(2),
    )

    pd.testing.assert_frame_equal(boot1, boot2)

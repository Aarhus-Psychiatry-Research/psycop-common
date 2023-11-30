import pytest

from psycop.projects.forced_admission_inpatient.utils.feature_name_to_readable import (
    feature_name_to_readable,
)


@pytest.mark.parametrize(
    ("feature_name", "expected_output"),
    [
        ("pred_skema_1_within_365_days_count_fallback_nan", "365-day count skema 1"),
        ("pred_age", "Age"),
    ],
)
def test_feature_name_to_readable(feature_name: str, expected_output: str):
    extracted_feature_name = feature_name_to_readable(feature_name)

    assert extracted_feature_name == expected_output

import pytest
from psycop.projects.t2d.utils.feature_name_to_readable import feature_name_to_readable


@pytest.mark.parametrize(
    ("feature_name", "expected_output"),
    [
        ("pred_hba1c_within_730_days_max_fallback_nan", "730-day max HbA1c"),
        ("pred_age", "Age"),
    ],
)
def test_feature_name_to_readable(feature_name: str, expected_output: str):
    pass
    extracted_feature_name = feature_name_to_readable(feature_name)

    assert extracted_feature_name == expected_output

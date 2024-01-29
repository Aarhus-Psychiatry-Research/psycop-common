import pandas as pd

from psycop.common.model_evaluation.binary.utils import auroc_by_group, sensitivity_by_group
from psycop.common.test_utils.str_to_df import str_to_df


def test_auroc_by_group():
    input_df = str_to_df(
        """id,y,y_hat_probs,
        1,1,0.9, # Good prediction
        1,1,0.8, # Good prediction
        1,1,0.7, # Good prediction
        1,0,0.6, # Good prediction
        1,0,0.5, # Good prediction
        2,1,0.7, # Good prediction
        2,1,0.2, # Bad prediction
        2,1,0.8, # Good prediction
        2,0,0.9, # Bad prediction
        2,0,0.1, # Good prediction
        3,1,0.5, # Bad prediction
        3,1,0.4, # Bad prediction
        3,1,0.3, # Bad prediction
        3,0,0.9, # Bad prediction
        3,0,0.8, # Bad prediction
        """
    )

    large_df = pd.concat([input_df for _ in range(10)])

    auroc_by_group_df = auroc_by_group(
        df=large_df, groupby_col_name="id", confidence_interval=True, n_bootstraps=10
    )

    assert auroc_by_group_df["auroc"].is_monotonic_decreasing
    assert auroc_by_group_df["n_in_bin"].to_list() == [50.0, 50.0, 50.0]
    assert auroc_by_group_df["ci_lower"].is_monotonic_decreasing
    assert auroc_by_group_df["ci_upper"].is_monotonic_decreasing


def test_sensitivity_by_group():
    input_df = str_to_df(
        """id,y,y_hat,
        1,1,1, # Good prediction
        1,1,1, # Good prediction
        1,1,1, # Good prediction
        2,1,0, # Bad prediction
        2,1,0, # Bad prediction
        2,1,0, # Bad prediction
        """
    )

    # Test using a categorical
    input_df["id"] = pd.Categorical(input_df["id"], ordered=True)

    large_df = pd.concat([input_df for _ in range(10)])

    output_df = sensitivity_by_group(
        df=large_df, groupby_col_name="id", confidence_interval=True, n_bootstraps=10
    )

    assert output_df["sensitivity"].to_list() == [1.0, 0.0]
    assert output_df["n_in_bin"].to_list() == [30.0, 30.0]
    assert output_df["ci_lower"].to_list() == [1.0, 0.0]
    assert output_df["ci_upper"].to_list() == [1.0, 0.0]

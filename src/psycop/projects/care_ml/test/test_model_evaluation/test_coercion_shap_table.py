import pandas as pd
import polars as pl
from psycop.common.test_utils.str_to_df import str_to_df

from care_ml.model_evaluation.figures.feature_importance.shap.shap_table import (
    get_top_i_shap_values_for_printing,
)


def test_get_top_2_shap_values_for_output():
    expected = str_to_df(
        """Rank,Feature,Mean absolute SHAP,
    1,Feature_3,0.75,
    2,Feature_2,0.3,
    """,
    )

    shap_long_df = str_to_df(
        """feature_name,feature_value,pred_time_index,shap_value
feature_1,1,0,0.1
feature_1,2,1,0.2
feature_2,1,0,0.2
feature_2,2,1,0.4
feature_3,1,0,0.5
feature_3,3,2,1.0
""",
    )

    computed = get_top_i_shap_values_for_printing(
        shap_long_df=pl.from_pandas(shap_long_df),
        i=2,
    ).to_pandas()

    # Compare with pandas
    pd.testing.assert_frame_equal(computed, expected, check_dtype=False)

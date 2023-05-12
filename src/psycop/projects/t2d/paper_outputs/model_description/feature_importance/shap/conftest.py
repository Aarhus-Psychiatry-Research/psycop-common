import polars as pl
import pytest
from psycop.common.test_utils.str_to_df import str_to_df


@pytest.fixture()
def shap_long_df() -> pl.DataFrame:
    pd_df = str_to_df(
        """feature_name,feature_value,pred_time_index,shap_value
feature_1,1,0,0.1
feature_1,2,1,0.2
feature_1,3,2,0.3
feature_2,1,0,0.2
feature_2,2,1,0.4
feature_2,3,2,0.6
feature_3,1,0,0.3
feature_3,2,1,0.6
feature_3,3,2,0.9
""",
    )

    return pl.from_pandas(pd_df)

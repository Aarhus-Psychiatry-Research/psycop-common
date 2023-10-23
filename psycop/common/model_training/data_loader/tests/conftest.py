import pandas as pd
import pytest

from psycop.common.test_utils.str_to_df import str_to_df


@pytest.fixture()
def base_feature_df() -> pd.DataFrame:
    return str_to_df(
        """prediction_time_uuid,feature_name_1,dw_ek_borger,timestamp
x_2010,0,x,2010-01-01
x_2011,0,x,2011-01-01
y_2010,1,y,2010-01-01
y_2011,1,y,2021-01-01""",
    )


@pytest.fixture()
def feature_df_same_order_uuids() -> pd.DataFrame:
    return str_to_df(
        """prediction_time_uuid,feature_name_2,dw_ek_borger,timestamp
x_2010,2,x,2010-01-01
x_2011,2,x,2011-01-01
y_2010,3,y,2010-01-01
y_2011,3,y,2021-01-01""",
    )


@pytest.fixture()
def feature_df_different_order_uuids() -> pd.DataFrame:
    return str_to_df(
        """prediction_time_uuid,feature_name_2,dw_ek_borger,timestamp
y_2010,3,y,2010-01-01
x_2010,2,x,2010-01-01
x_2011,2,x,2011-01-01
y_2011,3,y,2021-01-01""",
    )


@pytest.fixture()
def feature_df_different_split() -> pd.DataFrame:
    return str_to_df(   
            """prediction_time_uuid,feature_name_1,dw_ek_borger,timestamp
z_2010,0,x,2010-01-01
z_2011,0,x,2011-01-01
v_2010,1,y,2010-01-01
v_2011,1,y,2021-01-01""",
    )
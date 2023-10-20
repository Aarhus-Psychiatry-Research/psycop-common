from psycop.common.test_utils.str_to_df import str_to_df

## write test for hashing of uuids

## write test for merging feature sets


def test_uuid_match():
    str_to_df(
        """uuid,feature_name_1
        xx_xy_2010,0
        xx_xy_2011,0
        xx_yy_2010,1,
        xx_yy_2011,1""",
    )

    str_to_df(
        """uuid,feature_name_2
    xx_xy_2010,2
    xx_xy_2011,2
    xx_yy_2010,3,
    xx_yy_2011,3""",
    )

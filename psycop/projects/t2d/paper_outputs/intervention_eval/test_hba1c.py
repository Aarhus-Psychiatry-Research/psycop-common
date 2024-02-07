import datetime as dt

import polars as pl

from psycop.common.test_utils.str_to_df import str_to_pl_df
from psycop.projects.t2d.paper_outputs.intervention_eval.hba1c import (
    time_from_first_pos_pred_to_next_hba1c,
)


def test_time_from_first_pos_pred_to_next_hba1c():
    pos_preds = str_to_pl_df(
        """
                                  patient_id,pred_timestamps,
                                  1,2020-01-01, # First pos pred
                                  1,2021-01-01, # Prediction times after first pos pred
                                  2,2020-01-02, # Filtering if pred_timestamps == first pos pred
                                  """
    ).lazy()

    hba1cs = str_to_pl_df(
        """
        patient_id,timestamp,
        1,2020-01-02
        1,2020-01-03
        2,2019-01-01 # HbA1c before first pos pred should be excluded
        2,2020-01-02
    """
    ).lazy()

    expected = pl.DataFrame(
        {"patient_id": [1, 2], "min_time_to_next_hba1c": [dt.timedelta(days=1), dt.timedelta(0)]}
    ).sort(by="patient_id", descending=True)

    actual = (
        time_from_first_pos_pred_to_next_hba1c(pos_preds=pos_preds, hba1cs=hba1cs)
        .sort(by="patient_id", descending=True)
        .collect()
    )

    assert actual.frame_equal(expected, null_equal=True)

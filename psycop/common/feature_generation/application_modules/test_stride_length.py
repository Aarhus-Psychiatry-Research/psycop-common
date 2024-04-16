from psycop.common.cohort_definition import FilteredPredictionTimeBundle
from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
        T2DCohortDefiner,
    )
from timeseriesflattener import ValueFrame

from psycop.common.feature_generation.loaders.raw.load_structured_sfi import systolic_blood_pressure
import datetime as dt

import numpy as np
from timeseriesflattener import MaxAggregator, PredictionTimeFrame, PredictorSpec
from timeseriesflattener import Flattener
from psycop.common.global_utils.cache import mem

@mem.cache
def get_cohort(cache_version: int = 1) -> FilteredPredictionTimeBundle:
    return T2DCohortDefiner().get_filtered_prediction_times_bundle()

@mem.cache
def get_valueframe(cache_version: int = 1) -> ValueFrame:
    return ValueFrame(
                init_df=systolic_blood_pressure(),
                entity_id_col_name="dw_ek_borger",
                value_timestamp_col_name="timestamp",
            )

if __name__ == "__main__":
    cohort = get_cohort()

    value_frame = get_valueframe()
    
    predictor_specs = [
        PredictorSpec(
            value_frame=value_frame,
            lookbehind_distances=[dt.timedelta(days=i) for i in [1095]],
            aggregators=[MaxAggregator()],
            fallback=np.nan,
        )
    ]

    cohort_df = cohort.prediction_times.frame

    for i in [180]:
        start_time = dt.datetime.now()

        result = Flattener(
            predictiontime_frame=PredictionTimeFrame(
                init_df=cohort_df, entity_id_col_name="dw_ek_borger", timestamp_col_name="timestamp"
            )
        ).aggregate_timeseries(specs=predictor_specs, timedelta_days=i,)

        end_time = dt.datetime.now()
        print(f"Flattening with stride_length of {i} days tooks {end_time - start_time}")
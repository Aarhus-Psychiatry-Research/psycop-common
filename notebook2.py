import datetime as dt

import numpy as np
from timeseriesflattener import (
    MaxAggregator,
    MinAggregator,
    OutcomeSpec,
    PredictionTimeFrame,
    PredictorSpec,
    ValueFrame,
)

from psycop.common.feature_generation.loaders.raw.load_diagnoses import essential_hypertension
from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    T2DCohortDefiner,
)

predictor_specs = [
    PredictorSpec(
        value_frame=ValueFrame(
            init_df=essential_hypertension(),
            entity_id_col_name="dw_ek_borger",
            value_timestamp_col_name="timestamp",
        ),
        lookbehind_distances=[dt.timedelta(days=i) for i in range(1, 730)],
        aggregators=[MaxAggregator()],
        fallback=np.nan,
    )
]

if __name__ == "__main__":
    main()

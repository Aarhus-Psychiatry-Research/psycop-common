import datetime as dt
import logging
from dataclasses import KW_ONLY, dataclass

import numpy as np

from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    chronic_lung_disease,
    f0_disorders,
    f1_disorders,
    f2_disorders,
    f3_disorders,
    f4_disorders,
    f5_disorders,
    f6_disorders,
    f7_disorders,
    f8_disorders,
    f9_disorders,
    ischemic_stroke,
)
from psycop.common.feature_generation.loaders.raw.load_lab_results import hba1c, ldl
from psycop.common.feature_generation.loaders.raw.load_medications import (
    top_10_weight_gaining_antipsychotics,
)
from psycop.common.feature_generation.loaders.raw.load_procedures import cabg, pad, pci
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    height_in_cm,
    smoking_categorical,
    smoking_continuous,
    systolic_blood_pressure,
    weight_in_kg,
)
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    CVDCohortDefiner,
)
from psycop.timeseriesflattener.src.timeseriesflattenerv2.aggregators import (
    CountAggregator,
    MaxAggregator,
    MeanAggregator,
    MinAggregator,
)
from psycop.timeseriesflattener.src.timeseriesflattenerv2.feature_specs import (
    PredictionTimeFrame,
    PredictorSpec,
    ValueFrame,
)
from psycop.timeseriesflattener.src.timeseriesflattenerv2.flattener import Flattener

log = logging.getLogger(__file__)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/&m/%d %H:%M:%S",
    )
    lookbehind_distances = [dt.timedelta(days=days) for days in [90, 365, 730]]
    aggregators = [MeanAggregator(), MaxAggregator(), CountAggregator(), MinAggregator()]

    filtered_prediction_times = (
        CVDCohortDefiner.get_filtered_prediction_times_bundle()
        .prediction_times.frame.rename({"timestamp": "pred_timestamp"})
        .lazy()
    )

    flattener = Flattener(
        predictiontime_frame=PredictionTimeFrame(
            init_df=filtered_prediction_times, entity_id_col_name="dw_ek_borger"
        ),
        lazy=False,
    )

    feature_categories = {
        1: {
            "ldl": ldl,
            "smoking_continuous": smoking_continuous,
            "smoking_categorical": smoking_categorical,
        },
        2: {
            "pci": pci,
            "cabg": cabg,
            "ischemic_stroke": ischemic_stroke,
            "hba1c": hba1c,
            "chronic_lung_disease": chronic_lung_disease,
        },
        3: {
            "f0": f0_disorders,
            "f1": f1_disorders,
            "f2": f2_disorders,
            "f3": f3_disorders,
            "f4": f4_disorders,
            "f5": f5_disorders,
            "f6": f6_disorders,
            "f7": f7_disorders,
            "f8": f8_disorders,
            "f9": f9_disorders,
            "top_10_weight_gaining": top_10_weight_gaining_antipsychotics,
        },
    }

    specs = [
        PredictorSpec(
            value_frame=ValueFrame(
                init_df=loader().rename({"value": feature_name}, axis=1),
                value_col_name=feature_name,
                value_timestamp_col_name="timestamp",
            ),
            lookbehind_distances=lookbehind_distances,
            aggregators=aggregators,
            fallback=np.nan,
        )
        for category, spec_dict in feature_categories.items()
        for feature_name, loader in spec_dict.items()
    ]

    log.info(f"Starting flattening with {len(aggregators)} aggregators, {len(lookbehind_distances)} lookbehinds and {len(specs)} specs")
    df = flattener.aggregate_timeseries(specs=specs).df

    log.info("Finished flattening")

    pass

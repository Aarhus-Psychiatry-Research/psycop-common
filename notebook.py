import marimo

__generated_with = "0.3.12"
app = marimo.App()


@app.cell
def __():
    from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
        T2DCohortDefiner,
    )

    cohort = T2DCohortDefiner().get_filtered_prediction_times_bundle()
    return T2DCohortDefiner, cohort


@app.cell
def __():
    from timeseriesflattener import ValueFrame

    from psycop.common.feature_generation.loaders.raw.load_structured_sfi import systolic_blood_pressure

    value_frame = ValueFrame(
                init_df=systolic_blood_pressure(),
                entity_id_col_name="dw_ek_borger",
                value_timestamp_col_name="timestamp",
            )
    return ValueFrame, systolic_blood_pressure, value_frame


@app.cell
def __(value_frame):
    import datetime as dt

    import numpy as np
    from timeseriesflattener import MaxAggregator, PredictionTimeFrame, PredictorSpec

    predictor_specs = [
        PredictorSpec(
            value_frame=value_frame,
            lookbehind_distances=[dt.timedelta(days=i) for i in [365, 730, 1095]],
            aggregators=[MaxAggregator()],
            fallback=np.nan,
        )
    ]
    return (
        MaxAggregator,
        PredictionTimeFrame,
        PredictorSpec,
        dt,
        np,
        predictor_specs,
    )


@app.cell
def __(PredictionTimeFrame, cohort, dt, predictor_specs):
    from timeseriesflattener import Flattener

    cohort_df = cohort.prediction_times.frame

    for i in [180]:
        start_time = dt.datetime.now()

        result2 = Flattener(
            predictiontime_frame=PredictionTimeFrame(
                init_df=cohort_df, entity_id_col_name="dw_ek_borger", timestamp_col_name="timestamp"
            )
        ).aggregate_timeseries(specs=predictor_specs, timedelta_days=i,)

        end_time = dt.datetime.now()
        print(f"Flattening with stride_length of {i} days tooks {end_time - start_time}")
    return Flattener, cohort_df, end_time, i, result2, start_time


@app.cell
def __(result2):
    result2.df.collect()
    return


if __name__ == "__main__":
    app.run()

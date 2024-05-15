import marimo

__generated_with = "0.4.0"
app = marimo.App(width="full")


@app.cell
def __():
    from psycop.common.feature_generation.loaders.raw.load_lab_results import ldl
    import polars as pl
    return ldl, pl


@app.cell
def __(mo):
    mo.md(rf"")
    return


@app.cell
def __(pl):
    from psycop.common.feature_generation.loaders.raw.load_visits import (
        physical_visits,
    )

    visits = pl.from_pandas(physical_visits())
    return physical_visits, visits


@app.cell
def __(ldl, pl):
    ldl_timestamps = (
        pl.from_pandas(ldl())
        .drop("value")
        .with_columns(pl.col("timestamp").alias("latest_ldl_timestamp"))
    )
    ldl_timestamps
    return ldl_timestamps,


@app.cell
def __(ldl_timestamps, visits):
    from timeseriesflattener import (
        Flattener,
        PredictionTimeFrame,
        PredictorSpec,
        ValueFrame,
    )

    from timeseriesflattener.aggregators import LatestAggregator

    import datetime as dt

    with_latest_ldl = Flattener(
        predictiontime_frame=PredictionTimeFrame(
            init_df=visits,
            entity_id_col_name="dw_ek_borger",
            timestamp_col_name="timestamp",
        )
    ).aggregate_timeseries(
        specs=[
            PredictorSpec(
                value_frame=ValueFrame(
                    init_df=ldl_timestamps,
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookbehind_distances=[dt.timedelta(days=365 * 10)],
                aggregators=[LatestAggregator(timestamp_col_name="timestamp")],
                fallback=dt.datetime(1979, 1, 1),
            )
        ]
    )
    return (
        Flattener,
        LatestAggregator,
        PredictionTimeFrame,
        PredictorSpec,
        ValueFrame,
        dt,
        with_latest_ldl,
    )


@app.cell
def __(pl, with_latest_ldl):
    collected_df = with_latest_ldl.df.collect()
    plot_df = collected_df.with_columns(
        latest_ldl_age_days=(
            pl.col("timestamp")
            - pl.col(
                "pred_latest_ldl_timestamp_within_0_to_3650_days_latest_fallback_1979-01-01 00:00:00"
            )
        ).dt.total_days()
    )
    plot_df
    return collected_df, plot_df


@app.cell
def __(plot_df):
    import plotnine as pn

    plot = (
        pn.ggplot(data=plot_df)
        + pn.stat_ecdf(mapping=pn.aes(x="days_after_ldl"))
        + pn.scale_color_brewer(type="qual", palette=2)
        + pn.coord_cartesian(
            xlim=(
                0,
                int(365 * 5),
            )
        )
        + pn.ylab("Proportion with at least 1 LDL measurement")
        + pn.xlab("Days from visit to LDL measurement")
        + pn.scale_x_continuous(expand=(0, 0))
        + pn.scale_y_continuous(expand=(0, 0))
        + pn.theme_bw()
        + pn.theme(legend_position="bottom")
        + pn.labs(color="")
    )
    return plot, pn


@app.cell
def __(plot_df):
    n = len(plot_df)
    return n,


@app.cell
def __():
    import marimo as mo

    max_ldl_age_days = mo.ui.slider(1, 730, value=365, step=1, show_value=True)
    max_ldl_age_days
    return max_ldl_age_days, mo


@app.cell
def __(max_ldl_age_days, mo, n, pl, plot_df):
    n_needing_ldl_measurement = len(
        plot_df.filter(pl.col("latest_ldl_age_days") > max_ldl_age_days.value)
    )

    mo.md(
        f"Out of {n} visits, {n_needing_ldl_measurement} ({round(n_needing_ldl_measurement / n * 100, 0)}%) could use a new LDL measurement"
    )
    return n_needing_ldl_measurement,


if __name__ == "__main__":
    app.run()

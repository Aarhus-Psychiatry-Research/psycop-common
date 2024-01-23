from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn
import polars as pl
from timeseriesflattener.aggregation_fns import latest
from timeseriesflattener.feature_specs.single_specs import OutcomeSpec
from timeseriesflattener.flattened_dataset import TimeseriesFlattener

from psycop.common.feature_generation.loaders.raw.load_lab_results import (
    fasting_p_glc,
    hba1c,
    ogtt,
    unscheduled_p_glc,
)
from psycop.common.model_evaluation.binary.time.timedelta_data import get_timedelta_series
from psycop.common.model_training.preprocessing.pre_split.processors.row_filter import (
    PreSplitRowFilter,
)
from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline
from psycop.projects.t2d.utils.pipeline_objects import T2DPipelineRun


def get_eligible_prediction_times_for_pipeline(run: T2DPipelineRun) -> pd.DataFrame:
    col_names = run.inputs.cfg.data.col_name

    columns_to_keep = (
        col_names.id,
        col_names.pred_timestamp,
        col_names.outcome_timestamp,
        col_names.exclusion_timestamp,
        col_names.age,
    )

    flattened_dataset = (
        pl.concat([run.inputs.get_flattened_split_as_lazyframe(split=split) for split in ("test",)])
        .select(columns_to_keep)
        .collect()
    )

    cfg = run.inputs.cfg
    cfg.preprocessing.pre_split.Config.allow_mutation = True
    cfg.preprocessing.pre_split.lookbehind_combination = [365]

    eligible_for_pipeline = PreSplitRowFilter(
        data_cfg=cfg.data, pre_split_cfg=cfg.preprocessing.pre_split
    ).run_filter(dataset=flattened_dataset.to_pandas())

    return eligible_for_pipeline


class AbstractPlot(ABC):
    @abstractmethod
    def get_dataset(self, run: T2DPipelineRun) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _create_plot(self, df: pl.DataFrame, run: T2DPipelineRun) -> pn.ggplot:
        raise NotImplementedError

    @abstractmethod
    def get_plot(self, run: T2DPipelineRun) -> pn.ggplot:
        raise NotImplementedError


class MeasurementsWithinLookaheadPlot(AbstractPlot):
    def __init__(self):
        pass

    def get_dataset(self, run: T2DPipelineRun) -> pl.DataFrame:
        prediction_times_eligible_for_pipeline = get_eligible_prediction_times_for_pipeline(run=run)

        flattener = TimeseriesFlattener(
            prediction_times_df=prediction_times_eligible_for_pipeline[
                ["dw_ek_borger", "timestamp"]
            ].reset_index(drop=True),
            timestamp_col_name="timestamp",
            entity_id_col_name="dw_ek_borger",
            n_workers=1,
            outcome_col_name_prefix="eval",
            drop_pred_times_with_insufficient_look_distance=False,
        )

        lookahead_days = (
            get_best_eval_pipeline().inputs.cfg.preprocessing.pre_split.min_lookahead_days
        )

        hba1c_timestamps = hba1c()
        hba1c_timestamps["value"] = hba1c_timestamps["timestamp"]

        any_lab_result_dfs = [hba1c_timestamps, unscheduled_p_glc(), fasting_p_glc(), ogtt()]
        any_lab_result_timestamps = pd.concat(any_lab_result_dfs)
        any_lab_result_timestamps["value"] = any_lab_result_timestamps["timestamp"]

        specs = [
            OutcomeSpec(
                feature_base_name="hba1c",
                timeseries_df=hba1c_timestamps,
                aggregation_fn=latest,
                fallback=np.nan,
                incident=False,
                lookahead_days=lookahead_days,
            ),
            OutcomeSpec(
                feature_base_name="t2d_lab_result",
                timeseries_df=any_lab_result_timestamps,
                aggregation_fn=latest,
                fallback=np.nan,
                incident=False,
                lookahead_days=lookahead_days,
            ),
        ]
        flattener.add_spec(spec=specs)  # type: ignore

        flattened = flattener.get_df()

        eval_ds = run.pipeline_outputs.get_eval_dataset()

        results_df = pd.DataFrame(
            {
                "prediction_time_uuid": eval_ds.pred_time_uuids,
                "y": eval_ds.y,
                "y_hat": eval_ds.get_predictions_for_positive_rate(run.paper_outputs.pos_rate)[0],
            }
        )

        plot_df = results_df.merge(flattened, how="left", on="prediction_time_uuid", validate="1:1")

        for feature_name in ("t2d_lab_result", "hba1c"):
            flattened_col_name = f"outc_{feature_name}_within_1825_days_latest_fallback_nan"
            plot_df_col_name = f"years_until_last_{feature_name}"

            plot_df[plot_df_col_name] = get_timedelta_series(
                df=plot_df,
                direction="t2-t1",
                bin_unit="Y",
                t2_col_name=flattened_col_name,
                t1_col_name="timestamp",
            )

            plot_df[plot_df_col_name] = plot_df[plot_df_col_name].fillna(9999)

        return pl.from_dataframe(plot_df)

    def _create_plot(self, df: pl.DataFrame, run: T2DPipelineRun) -> pn.ggplot:
        df = (
            df.select(
                [
                    "prediction_time_uuid",
                    "y",
                    "y_hat",
                    "years_until_last_t2d_lab_result",
                    "years_until_last_hba1c",
                ]
            )
            .with_columns(
                prediction=pl.when((pl.col("y") == 0) & (pl.col("y_hat") == 1))
                .then("False positive")
                .when((pl.col("y") == 0) & (pl.col("y_hat") == 0))
                .then("True negative")
                .when((pl.col("y") == 1) & (pl.col("y_hat") == 1))
                .then("True positive")
                .when((pl.col("y") == 1) & (pl.col("y_hat") == 0))
                .then("False negative")
            )
            .rename(
                {
                    "years_until_last_t2d_lab_result": "HbA1c, OGTT, fasting p-Glc or unscheduled p-Glc",
                    "years_until_last_hba1c": "HbA1c",
                }
            )
            .melt(
                id_vars=["prediction_time_uuid", "prediction"],
                value_vars=["HbA1c, OGTT, fasting p-Glc or unscheduled p-Glc", "HbA1c"],
            )
        ).filter(pl.col("prediction") == "False positive")

        plot = (
            pn.ggplot(data=df)
            + pn.stat_ecdf(mapping=pn.aes(x="value", color="variable"))
            + pn.scale_color_brewer(type="qual", palette=2)
            + pn.facet_wrap("prediction", nrow=2)
            + pn.coord_cartesian(
                xlim=(0, int(run.inputs.cfg.preprocessing.pre_split.min_lookahead_days / 365))
            )
            + pn.ylab("Last measurement in lookahead window")
            + pn.xlab("Years since prediction time")
            + pn.scale_x_continuous(expand=(0, 0))
            + pn.scale_y_continuous(expand=(0, 0))
            + pn.theme_bw()
            + pn.theme(legend_position="bottom")
            + pn.labs(color="")
        )

        plot.save(Path() / "test.png")

        return plot

    def get_plot(self, run: T2DPipelineRun) -> pn.ggplot:
        df = self.get_dataset(run=run)

        plot = self._create_plot(df=df, run=run)
        plot.draw()

        return plot


if __name__ == "__main__":
    pipeline = get_best_eval_pipeline()
    plot = MeasurementsWithinLookaheadPlot().get_plot(run=get_best_eval_pipeline())
    size = (6.5, 8)

    plot.save(
        pipeline.paper_outputs.paths.figures / "t2d_last_diabetes_measurement.png",
        width=size[0],
        height=size[1],
        dpi=600,
    )

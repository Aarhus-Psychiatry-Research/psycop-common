from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn
import polars as pl
from psycop.common.feature_generation.loaders.raw.load_lab_results import hba1c
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_timedelta_series,
)
from psycop.common.model_training.preprocessing.pre_split.processors.row_filter import (
    PreSplitRowFilter,
)
from psycop.projects.t2d.paper_outputs.selected_runs import BEST_EVAL_PIPELINE
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun
from timeseriesflattener.feature_spec_objects import OutcomeSpec
from timeseriesflattener.flattened_dataset import TimeseriesFlattener
from timeseriesflattener.resolve_multiple_functions import latest


def get_eligible_prediction_times_for_pipeline(run: PipelineRun) -> pd.DataFrame:
    col_names = run.inputs.cfg.data.col_name

    columns_to_keep = (
        col_names.id,
        col_names.pred_timestamp,
        col_names.outcome_timestamp,
        col_names.exclusion_timestamp,
        col_names.age,
    )

    flattened_dataset = (
        pl.concat(
            [
                run.inputs.get_flattened_split_as_lazyframe(split=split)
                for split in ("test",)
            ],
        )
        .select(columns_to_keep)
        .collect()
    )

    cfg = run.inputs.cfg
    cfg.preprocessing.pre_split.Config.allow_mutation = True
    cfg.preprocessing.pre_split.lookbehind_combination = [365]

    eligible_for_pipeline = PreSplitRowFilter(
        data_cfg=cfg.data,
        pre_split_cfg=cfg.preprocessing.pre_split,
    ).run_filter(dataset=flattened_dataset.to_pandas())

    return eligible_for_pipeline


class AbstractPlot(ABC):
    @abstractmethod
    def get_dataset(self, run: PipelineRun) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _create_plot(self, df: pd.DataFrame) -> pn.ggplot:
        raise NotImplementedError

    @abstractmethod
    def get_plot(self, run: PipelineRun) -> pn.ggplot:
        raise NotImplementedError


class HbA1cWithinLookaheadPlot(AbstractPlot):
    def __init__(self):
        pass

    def get_dataset(self, run: PipelineRun) -> pd.DataFrame:
        prediction_times_eligible_for_pipeline = (
            get_eligible_prediction_times_for_pipeline(
                run=run,
            )
        )

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
            BEST_EVAL_PIPELINE.inputs.cfg.preprocessing.pre_split.min_lookahead_days
        )

        hba1c_timestamps = hba1c()
        hba1c_timestamps["value"] = hba1c_timestamps["timestamp"]

        spec = OutcomeSpec(
            feature_name="hba1c",
            values_df=hba1c_timestamps,
            resolve_multiple_fn=latest,
            fallback=np.nan,
            entity_id_col_name="dw_ek_borger",
            incident=False,
            lookahead_days=lookahead_days,
        )
        flattener.add_spec(spec=spec)

        flattened = flattener.get_df()

        eval_ds = run.pipeline_outputs.get_eval_dataset()

        results_df = pd.DataFrame(
            {
                "prediction_time_uuid": eval_ds.pred_time_uuids,
                "y": eval_ds.y,
                "y_hat": eval_ds.get_predictions_for_positive_rate(
                    run.paper_outputs.pos_rate,
                )[0],
            },
        )

        plot_df = results_df.merge(
            flattened,
            how="left",
            on="prediction_time_uuid",
            validate="1:1",
        )

        plot_df["years_until_last_hba1c"] = get_timedelta_series(
            df=plot_df,
            direction="t2-t1",
            bin_unit="Y",
            t2_col_name="outc_hba1c_within_1825_days_latest_fallback_nan",
            t1_col_name="timestamp",
        )

        plot_df["years_until_last_hba1c"] = plot_df["years_until_last_hba1c"].fillna(
            9999
        )

        return plot_df

    def _create_plot(self, df: pd.DataFrame, run: PipelineRun) -> pn.ggplot:
        plot = (
            pn.ggplot(data=df, mapping=pn.aes(x="years_until_last_hba1c"))
            + pn.stat_ecdf()
            + pn.coord_cartesian(
                xlim=(
                    0,
                    int(
                        run.inputs.cfg.preprocessing.pre_split.min_lookahead_days / 365
                    ),
                ),
            )
            + pn.ylab("Last HbA1c in lookahead window")
            + pn.xlab("Years since prediction time")
            + pn.scale_x_continuous(expand=(0, 0))
            + pn.scale_y_continuous(expand=(0, 0))
            + pn.theme_bw()
        )

        plot.save(Path(".") / "test.png")

        return plot

    def get_plot(self, run: PipelineRun) -> pn.ggplot:
        df = self.get_dataset(run=run)

        plot = self._create_plot(df=df, run=run)
        plot.draw()

        return plot


class Hba1cWithinLookaheadForFalsePositives(HbA1cWithinLookaheadPlot):
    def get_dataset(self, run: PipelineRun) -> pl.DataFrame:
        df = super().get_dataset(run)

        false_positives = pl.from_pandas(data=df).filter(
            (pl.col("y") == 0) & (pl.col("y_hat") == 1),
        )

        return false_positives

    def _create_plot(self, df: pd.DataFrame, run: PipelineRun) -> pn.ggplot:
        p = super()._create_plot(df, run)
        return p + pn.ggtitle("False positives")


class Hba1cWithinLookaheadForTrueNegatives(HbA1cWithinLookaheadPlot):
    def get_dataset(self, run: PipelineRun) -> pl.DataFrame:
        df = super().get_dataset(run)

        true_negatives = pl.from_pandas(data=df).filter(
            (pl.col("y") == 0) & (pl.col("y_hat") == 0),
        )

        return true_negatives

    def _create_plot(self, df: pd.DataFrame, run: PipelineRun) -> pn.ggplot:
        p = super()._create_plot(df, run)
        return p + pn.ggtitle("True negatives")


if __name__ == "__main__":
    pipeline = BEST_EVAL_PIPELINE
    false_positives = Hba1cWithinLookaheadForFalsePositives().get_plot(
        run=BEST_EVAL_PIPELINE
    )
    size = (5, 3)

    false_positives.save(
        pipeline.paper_outputs.paths.figures / "t2d_last_hba1c_false_positives.png",
        width=size[0],
        height=size[1],
        dpi=600,
    )

    true_negatives = Hba1cWithinLookaheadForTrueNegatives().get_plot(
        run=BEST_EVAL_PIPELINE
    )
    true_negatives.save(
        pipeline.paper_outputs.paths.figures / "t2d_last_hba1c_true_negatives.png",
        width=size[0],
        height=size[1],
        dpi=600,
    )

    pass

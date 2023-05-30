from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn
import polars as pl
from matplotlib.cbook import flatten
from psycop.common.feature_generation.application_modules.filter_prediction_times import (
    PredictionTimeFilterer,
)
from psycop.common.feature_generation.loaders.raw.load_lab_results import hba1c
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_timedelta_series,
)
from psycop.common.model_training.preprocessing.pre_split.full_processor import (
    FullProcessor,
)
from psycop.common.model_training.preprocessing.pre_split.processors.row_filter import (
    PreSplitRowFilter,
)
from psycop.projects.t2d.feature_generation.eligible_prediction_times.loader import (
    get_eligible_prediction_times_as_pandas,
    get_eligible_prediction_times_as_polars,
)
from psycop.projects.t2d.paper_outputs.config import PN_THEME
from psycop.projects.t2d.paper_outputs.selected_runs import BEST_EVAL_PIPELINE
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun
from timeseriesflattener.feature_spec_objects import OutcomeSpec
from timeseriesflattener.flattened_dataset import TimeseriesFlattener
from timeseriesflattener.resolve_multiple_functions import latest
from wasabi import Printer


def get_pipeline_eligible_prediction_times(run: PipelineRun) -> pd.DataFrame:
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
                for split in ("train", "test", "val")
            ]
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
        prediction_times_eligible_for_pipeline = get_pipeline_eligible_prediction_times(
            run=run
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
                    run.paper_outputs.pos_rate
                )[0],
            }
        )

        plot_df = results_df.merge(
            flattened, how="left", on="prediction_time_uuid", validate="1:1"
        )

        plot_df["days_until_hba1c"] = get_timedelta_series(
            df=plot_df,
            direction="t2-t1",
            bin_unit="D",
            t2_col_name="outc_hba1c_within_1825_days_latest_fallback_nan",
            t1_col_name="timestamp",
        )

        plot_df["days_until_hba1c"] = plot_df["days_until_hba1c"].fillna(9999)

        return plot_df

    def _create_plot(self, df: pd.DataFrame, run: PipelineRun) -> pn.ggplot:
        plot = (
            pn.ggplot(data=df, mapping=pn.aes(x="days_until_hba1c"))
            + pn.stat_ecdf()
            + pn.coord_cartesian(
                xlim=(0, run.inputs.cfg.preprocessing.pre_split.min_lookahead_days)
            )
            + pn.ylab("No further HbA1c measurements \nwithin lookahead window")
            + pn.xlab("Days since prediction time")
            + PN_THEME
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

        false_positives = pl.from_pandas(df).filter(
            (pl.col("y") == 0) & (pl.col("y_hat") == 1)
        )

        return false_positives


class Hba1cWithinLookaheadForTrueNegatives(HbA1cWithinLookaheadPlot):
    def get_dataset(self, run: PipelineRun) -> pl.DataFrame:
        df = super().get_dataset(run)

        true_negatives = pl.from_pandas(df).filter(
            (pl.col("y") == 0) & (pl.col("y_hat") == 0)
        )

        return true_negatives


if __name__ == "__main__":
    plot = HbA1cWithinLookaheadPlot().get_plot(run=BEST_EVAL_PIPELINE)

from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.restraint.evaluation.utils import (
    read_eval_df_from_disk,
)

import datetime
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import PsycopMlflowRun
from psycop.common.model_evaluation.binary.global_performance.roc_auc import bootstrap_roc


def plotnine_temporal_validation(df: pl.DataFrame, title: str = "Temporal Stability") -> pn.ggplot:

    df = df.with_columns(
        pl.col("train_end_date").dt.year().cast(pl.Utf8).alias("train_end_year"),
        (pl.col("start_date") - (pl.col("train_end_date"))).alias("since_train_end"),
    )

    def format_datetime(x: Sequence[datetime.datetime]) -> Sequence[str]:  # type: ignore
        return [f"{d.year-2000}-{d.year+1-2000}" for d in x]

    def format_datetime_v2(x: Sequence[datetime.datetime]) -> Sequence[str]:  # type: ignore
        return [f"2014-{d.year}" for d in x]

    df = df.with_columns(
        pl.Series(name="year", values=format_datetime(df["end_date"])),  # type: ignore
        pl.Series(name="train_interval", values=format_datetime_v2(df["train_end_date"])),  # type: ignore
    )
      
    p = (
        pn.ggplot(df, pn.aes(
                    x="year",
                    y="performance",
                    color="train_interval",
                    group="train_interval",
                ))
        + pn.scale_color_ordinal()
        + pn.geom_line()
        + pn.geom_point(size=2)
        + pn.geom_errorbar(pn.aes(x="year", ymin="lower_ci", ymax="upper_ci"))
        + pn.labs(x="Validation year", y="AUROC", title=title, color="Training years")
        + pn.ylim(0.75, 1)
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(size=15, angle=45, ha="right"),
            axis_text_y=pn.element_text(size=15),
            panel_grid_minor=pn.element_blank(),
            text=(pn.element_text(family="Times New Roman")),
            axis_title=pn.element_text(size=22),
            plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
        )
        #+ pn.scale_x_discrete()
        #+ pn.scale_fill_manual(values=["#669BBC", "#A8C686", "#F3A712"])
    )

    return p

@dataclass
class TemporalRunPerformance:
    run: PsycopMlflowRun
    bootstrapped_performances: Sequence[float]
    performance: float
    lower_ci: float
    upper_ci: float
    train_end_date: datetime.datetime
    start_date: datetime.datetime
    end_date: datetime.datetime

    def to_dataframe(self) -> pl.DataFrame:
        # Convert estimates to a list
        # Create a dictionary with the data
        data = {
            "performance": [self.performance],
            "bootstrapped_estimates": [self.bootstrapped_performances],
            "lower_ci": [self.lower_ci],
            "upper_ci": [self.upper_ci],
            "start_date": [self.start_date],
            "end_date": [self.end_date],
            "train_end_date": [self.train_end_date],
        }

        # Create the DataFrame
        df = pl.DataFrame(data)

        return df


@shared_cache().cache
def run_perf_from_disk(run: PsycopMlflowRun, n_bootstraps: int) -> TemporalRunPerformance:
    config = run.get_config()

    eval_frame =  read_eval_df_from_disk(config.retrieve("logger.*.disk_logger.run_path"))

    y = eval_frame.to_pandas()["y"]
    y_hat_probs = eval_frame.to_pandas()["y_hat_prob"]
    _, aucs_bootstrapped, _ = bootstrap_roc(n_bootstraps=n_bootstraps, y=y, y_hat_probs=y_hat_probs)

    auc_mean = float(np.mean(aucs_bootstrapped))
    auc_se = np.std(aucs_bootstrapped) / np.sqrt(n_bootstraps)
    auc_ci = [auc_mean - 1.96 * auc_se, auc_mean + 1.96 * auc_se]

    # Get dates
    date_parser = lambda d: datetime.datetime.strptime(d, "%Y-%m-%d")  # noqa: E731
    train_end_date = date_parser(
        config.retrieve("trainer.training_preprocessing_pipeline.*.date_filter.threshold_date")
    )
    start_eval_date = date_parser(
        config.retrieve(
            "trainer.validation_preprocessing_pipeline.*.date_filter_start.threshold_date"
        )
    )
    end_eval_date = date_parser(
        config.retrieve(
            "trainer.validation_preprocessing_pipeline.*.date_filter_end.threshold_date"
        )
    )

    return TemporalRunPerformance(
        run=run,
        performance=auc_mean,
        lower_ci=auc_ci[0],
        upper_ci=auc_ci[1],
        train_end_date=train_end_date,
        start_date=start_eval_date,
        end_date=end_eval_date,
        bootstrapped_performances=aucs_bootstrapped.tolist(),
    )


def temporal_stability(
    runs: Sequence[PsycopMlflowRun], n_bootstraps: int
) -> Sequence[TemporalRunPerformance]:
    """Temporal stability model for T2D Extended."""
    return [run_perf_from_disk(run, n_bootstraps=n_bootstraps) for run in runs]



if __name__ == "__main__":
    save_dir = Path(__file__).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    runs = MlflowClientWrapper().get_runs_from_experiment("restraint_temporal_validation")

    xg_performances = temporal_stability(runs, n_bootstraps=1000)

    df = pl.concat(run.to_dataframe() for run in xg_performances)

    plotnine_temporal_validation(df).save(
        save_dir / "restraint_temporal_validation.png"
    )

from pathlib import Path

import plotnine as pn
import polars as pl
from confection import Config

# Set path to BaselineSchema for the run
## Load dataset with predictions after training
## Load validation dataset
## 1:1 join metadata cols to predictions
# Plot performance by
## Age
## Sex
## Calendar time
## Diagnosis type (scz or bp)
## Time to event
## Time from first visit
# Table of performance (sens, spec, ppv, f1) by threshold
## Confusion matrix at specified threshold
# Plot feature importance
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)
from psycop.common.model_training_v2.trainer.cross_validator_trainer import (
    CrossValidatorTrainer,
)
from psycop.common.model_training_v2.trainer.split_trainer import SplitTrainer
from psycop.projects.scz_bp.evaluation.model_performance.performance.performance_by_time_to_event import (
    scz_bp_plot_sensitivity_by_time_to_event,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_age import (
    scz_bp_auroc_by_age,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_calendar_time import (
    scz_bp_auroc_by_quarter,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_cyclic_time import (
    scz_bp_auroc_by_day_of_week,
    scz_bp_auroc_by_month_of_year,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_sex import (
    scz_bp_auroc_by_sex,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_time_from_first_visit import (
    scz_bp_auroc_by_time_from_first_contact,
)

cfg_path = OVARTACI_SHARED_DIR / "scz_bp" / "experiments" / "l1"
populate_baseline_registry()


def load_and_resolve_cfg(path: Path) -> BaselineSchema:
    cfg = Config().from_disk(path)
    cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg))
    return cfg_schema


def prediction_time_uuid_to_prediction_time(
    prediction_time_uuid_series: pl.Series,
) -> pl.Series:
    return (
        prediction_time_uuid_series.str.split("-")
        .list.slice(1)
        .list.join("-")
        .str.to_datetime()
    )


def scz_bp_df_to_eval_df(
    df: pl.DataFrame,
    y_hat_prop_col_name: str,
    y_col_name: str,
) -> EvalDataset:
    return EvalDataset(
        ids=df["dw_ek_borger"].to_pandas(),
        pred_time_uuids=df["prediction_time_uuid"].to_pandas(),
        pred_timestamps=df["timestamp"].to_pandas(),
        outcome_timestamps=df["meta_time_of_diagnosis"].to_pandas(),
        y=df[y_col_name].to_pandas(),
        y_hat_probs=df[y_hat_prop_col_name].to_pandas(),
        age=df["pred_age_in_years"].to_pandas(),
        is_female=df["pred_sex_female_layer_1"].to_pandas(),
        custom_columns={
            "scz_or_bp": df["meta_scz_or_bp_indicator"].to_pandas(),
            "first_visit": df["meta_first_visit"].to_pandas(),
        },
    )


def merge_pred_df_with_validation_df(
    pred_df: pl.DataFrame,
    validation_df: pl.DataFrame,
) -> pl.DataFrame:
    validation_df = validation_df.select(
        pl.col("^meta.*$"),
        "timestamp",
        "prediction_time_uuid",
    )
    return pred_df.join(validation_df, how="left", on="prediction_time_uuid")


class EvalConfigResolver:
    def __init__(self, path_to_cfg: Path):
        self.path = path_to_cfg
        self.schema = load_and_resolve_cfg(path=path_to_cfg)

        validation_df = self._load_validation_data_from_schema()
        pred_df = self._read_pred_df()
        self.outcome_col_name = pred_df.select(pl.col("^outc_.*$")).columns[0]

        self.df = merge_pred_df_with_validation_df(
            pred_df=pred_df,
            validation_df=validation_df,
        )
        self.eval_ds = scz_bp_df_to_eval_df(
            df=self.df,
            y_hat_prop_col_name=self.y_hat_prop_col_name,
            y_col_name=self.outcome_col_name,
        )

    def _load_validation_data_from_schema(self) -> pl.DataFrame:
        match self.schema.trainer:
            case CrossValidatorTrainer():
                self.y_hat_prop_col_name = "oof_y_hat_prob"
                return self.schema.trainer.training_data.load().collect()
            case SplitTrainer():
                self.y_hat_prop_col_name = "y_hat_prob"
                return self.schema.trainer.validation_data.load().collect()
            case _:
                raise ValueError(
                    f"Handler for {type(self.schema.trainer)} not implemented",
                )

    def _read_pred_df(self) -> pl.DataFrame:
        return pl.read_parquet(
            self.schema.project_info.experiment_path / "eval_df.parquet",
        )


def full_eval(run: EvalConfigResolver) -> list[pn.ggplot]:
    eval_ds = run.eval_ds

    age = scz_bp_auroc_by_age(eval_ds=eval_ds)
    sex = scz_bp_auroc_by_sex(eval_ds=eval_ds)
    time_from_first_visit = scz_bp_auroc_by_time_from_first_contact(eval_ds=eval_ds)

    dow = scz_bp_auroc_by_day_of_week(eval_ds=eval_ds)
    month = scz_bp_auroc_by_month_of_year(eval_ds=eval_ds)
    quarter = scz_bp_auroc_by_quarter(eval_ds=eval_ds)

    sens_time_to_event = scz_bp_plot_sensitivity_by_time_to_event(eval_ds=eval_ds)
    return [age, sex, time_from_first_visit, dow, month, quarter, sens_time_to_event]


if __name__ == "__main__":
    run = EvalConfigResolver(path_to_cfg=cfg_path / "config.cfg")

    full_eval(run)

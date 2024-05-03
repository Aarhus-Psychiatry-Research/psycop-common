import datetime as dt
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import polars.selectors as cs
from timeseriesflattener import Flattener
from timeseriesflattener import PredictionTimeFrame as FlattenerPredictionTimeFrame

from psycop.common.cohort_definition import PredictionTimeFrame
from psycop.common.feature_generation.application_modules.generate_feature_set import (
    ValueSpecification,
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.feature_generation.scz_bp_specify_features import (
    SczBpFeatureSpecifier,
)


@dataclass
class FeatureSets:
    no_prevalent_df: pl.DataFrame
    only_prevalent_df: pl.DataFrame


def get_scz_bp_project_info() -> ProjectInfo:
    return ProjectInfo(project_name="scz_bp", project_path=OVARTACI_SHARED_DIR / "scz_bp")


def sczbp_generate_feature_set(
    eligible_prediction_times: PredictionTimeFrame,
    feature_specs: Sequence[ValueSpecification],
    n_workers: int | None = None,
    step_size: dt.timedelta | None = None,
) -> pl.DataFrame:
    flattener = Flattener(
        predictiontime_frame=FlattenerPredictionTimeFrame(
            init_df=eligible_prediction_times.frame,
            entity_id_col_name=eligible_prediction_times.entity_id_col_name,
            timestamp_col_name=eligible_prediction_times.timestamp_col_name,
        ),
        n_workers=n_workers,
        compute_lazily=False,
    )
    return flattener.aggregate_timeseries(specs=feature_specs, step_size=step_size).df.collect()


def split_feature_set_by_prevalence(
    feat_set: pl.DataFrame, prevalent_col_name: str
) -> FeatureSets:
    """Split feature set into one with only prevalent, and one without prevalent prediction times.
    Replace outcome column values with 1s for the prevalent df
    """
    feat_set_only_prevalent = feat_set.filter(pl.col(prevalent_col_name) == 1).with_columns(
        cs.starts_with("outc_").replace(old=0, new=1)
    )

    feat_set_no_prevalent = feat_set.filter(pl.col(prevalent_prediction_time_col_name) == 0)
    return FeatureSets(
        no_prevalent_df=feat_set_no_prevalent, only_prevalent_df=feat_set_only_prevalent
    )


def generate_feature_set_split_by_prevalence(
    project_path: Path,
    feature_set_name: str,
    prevalent_prediction_time_col_name: str,
    eligible_prediction_times: PredictionTimeFrame,
    feature_specs: Sequence[ValueSpecification],
    step_size: None | dt.timedelta,
    n_workers: None | int
) -> None:
    t0 = time.time()

    # generate features with tsflattener
    flattener = Flattener(
    predictiontime_frame=FlattenerPredictionTimeFrame(
        init_df=eligible_prediction_times.frame,
        entity_id_col_name=eligible_prediction_times.entity_id_col_name,
        timestamp_col_name=eligible_prediction_times.timestamp_col_name,
    ),
    n_workers=n_workers,
    compute_lazily=False,
)
    feat_set = flattener.aggregate_timeseries(specs=feature_specs, step_size=step_size).df.collect()

    t = time.time()
    print(f"Feature generation took: {t - t0}. step_size: {step_size}, n_workers: {n_workers}")

    split_feature_sets = split_feature_set_by_prevalence(
        feat_set=feat_set, prevalent_col_name=prevalent_prediction_time_col_name
    )

    feature_set_dir = project_path / "flattened_datasets" / feature_set_name
    feature_set_dir.mkdir(parents=True, exist_ok=True)

    split_feature_sets.no_prevalent_df.write_parquet(feature_set_dir / "no_prevalent.parquet")
    split_feature_sets.only_prevalent_df.write_parquet(feature_set_dir / "only_prevalent.parquet")


if __name__ == "__main__":
    feature_set_name = "structured_predictors_only"
    project_info = get_scz_bp_project_info()
    prevalent_prediction_time_col_name = "meta_prevalent_within_0_to_7300_days_bool_fallback_0"

    pred_times = SczBpCohort.get_filtered_prediction_times_bundle().prediction_times
    specs = SczBpFeatureSpecifier().get_feature_specs(
        max_layer=4,
        lookbehind_days=[183, 365, 730]
    )

    generate_feature_set_split_by_prevalence(
        project_path=project_info.project_path,
        feature_set_name=feature_set_name,
        prevalent_prediction_time_col_name=prevalent_prediction_time_col_name,
        eligible_prediction_times=pred_times,
        feature_specs=specs,
        step_size=dt.timedelta(days=365),
        n_workers=10
    )

"""Main script for defining features/predictors used for predicting UVI"""

import datetime
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd
import polars as pl
import timeseriesflattener as ts
from timeseriesflattener.aggregators import HasValuesAggregator

from psycop.common.feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.uti.feature_generation.cohort_definition.uti_cohort_definer import (
    uti_pred_times,
)
from psycop.projects.uti.feature_generation.outcome_definition.uti_outcomes import (
    uti_outcomes,
    uti_postive_urine_sample_outcome_timestamps,
    uti_relevant_antibiotics_administrations_outcome_timestamps,
)


def get_uti_project_info() -> ProjectInfo:
    return ProjectInfo(project_name="uti", project_path=OVARTACI_SHARED_DIR / "uti")


# Lots of code for specifying schemas and object - not too relevant
def _init_uti_predictor(
    df_loader: Callable[[], pd.DataFrame],
    layer: str,
    fallback: float | int,
    aggregation_fns: Sequence[ts.aggregators.Aggregator],
    lookbehind_distances: Sequence[datetime.timedelta] = [
        datetime.timedelta(days=i) for i in [30, 365]
    ],
    column_prefix: str = "pred_layer_{}",
    entity_id_col_name: str = "dw_ek_borger",
    name_overwrite: str | None = None,
) -> ts.PredictorSpec:
    logging.info(f"Initialising {df_loader.__name__ if name_overwrite is None else name_overwrite}")
    return ts.PredictorSpec(
        value_frame=ts.ValueFrame(
            init_df=pl.from_pandas(df_loader()).rename(
                {"value": df_loader.__name__ if name_overwrite is None else name_overwrite}
            ),
            entity_id_col_name=entity_id_col_name,
        ),
        lookbehind_distances=lookbehind_distances,
        aggregators=aggregation_fns,
        fallback=fallback,
        column_prefix=column_prefix.format(layer),
    )


AnySpec = ts.PredictorSpec | ts.OutcomeSpec | ts.StaticSpec | ts.TimeDeltaSpec


@dataclass(frozen=True)
class ContinuousSpec:
    loader: Callable[[], pd.DataFrame]
    aggregation_fns: Sequence[ts.aggregators.Aggregator]
    fallback: float
    lookbehind_distances: Sequence[datetime.timedelta] = [
        datetime.timedelta(days=i) for i in [30, 365]
    ]
    name_overwrite: str | None = None


@dataclass(frozen=True)
class BooleanSpec:
    loader: Callable[[], pd.DataFrame]
    aggregation_fns: Sequence[ts.aggregators.Aggregator] = (HasValuesAggregator(),)
    fallback: float = 0.0
    lookbehind_distances: Sequence[datetime.timedelta] = [
        datetime.timedelta(days=i) for i in [30, 365]
    ]
    name_overwrite: str | None = None


@dataclass(frozen=True)
class CategoricalSpec:
    loader: Callable[[], pd.DataFrame]
    aggregation_fns: Sequence[ts.aggregators.Aggregator]
    fallback: float
    lookbehind_distances: Sequence[datetime.timedelta] = [
        datetime.timedelta(days=i) for i in [30, 365]
    ]
    name_overwrite: str | None = None


@dataclass(frozen=True)
class LayerSpecPair:
    layer: str
    spec: AnySpec | BooleanSpec | ContinuousSpec | CategoricalSpec


def _pair_to_spec(pair: LayerSpecPair) -> AnySpec:
    match pair.spec:
        case ContinuousSpec():
            return _init_uti_predictor(
                df_loader=pair.spec.loader,
                layer=pair.layer,
                fallback=pair.spec.fallback,
                lookbehind_distances=pair.spec.lookbehind_distances,
                aggregation_fns=pair.spec.aggregation_fns,
                name_overwrite=pair.spec.name_overwrite,
            )
        case BooleanSpec():
            return _init_uti_predictor(
                df_loader=pair.spec.loader,
                layer=pair.layer,
                fallback=pair.spec.fallback,
                lookbehind_distances=pair.spec.lookbehind_distances,
                aggregation_fns=pair.spec.aggregation_fns,
                name_overwrite=pair.spec.name_overwrite,
            )
        case CategoricalSpec():
            return _init_uti_predictor(
                df_loader=pair.spec.loader,
                layer=pair.layer,
                fallback=pair.spec.fallback,
                lookbehind_distances=pair.spec.lookbehind_distances,
                aggregation_fns=pair.spec.aggregation_fns,
                name_overwrite=pair.spec.name_overwrite,
            )
        case ts.PredictorSpec() | ts.OutcomeSpec() | ts.StaticSpec() | ts.TimeDeltaSpec():
            return pair.spec


# Main function for generating features
def uti_generate_features(
    outcomes: Literal["combined", "urine_samples", "antibiotics"] = "combined",
    lookahead_days: int = 2,
    feature_set_name: str = "uti_feature_set",
    antibiotics_window: tuple[int, int] = (-1, 5),
    save: bool = True,
) -> pl.DataFrame | None:
    # Loads the specified outcomes
    match outcomes:
        case "combined":
            outcome_df = uti_outcomes(antibiotics_window).assign(value=1)
        case "urine_samples":
            outcome_df = uti_postive_urine_sample_outcome_timestamps().assign(value=1)
        case "antibiotics":
            outcome_df = uti_relevant_antibiotics_administrations_outcome_timestamps().assign(
                value=1
            )

    # Define features
    feature_layers = {
        "basic": [
            # Add outcome
            ts.OutcomeSpec(
                value_frame=ts.ValueFrame(
                    init_df=outcome_df,
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookahead_distances=[datetime.timedelta(days=lookahead_days)],
                aggregators=[ts.MaxAggregator()],
                fallback=0,
                column_prefix="outc_uti",
            ),
            # Add sex feature
            ts.StaticSpec(
                value_frame=ts.StaticFrame(init_df=sex_female(), entity_id_col_name="dw_ek_borger"),
                fallback=0,
                column_prefix="pred",
            ),
            # Add age feature (in days)
            ts.TimeDeltaSpec(
                init_frame=ts.TimestampValueFrame(
                    birthdays(),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="date_of_birth",
                ),
                column_prefix="pred",
                fallback=0,
                output_name="age",
            ),
            # Add earlier uti-relevant antibiotics as a feature/preictors. Creates two seperate features looking back 30 and 365 days from the prediction timestamp. Features are coded as boolean (1=has been amdinistered antibiotics in this time window, 0=has not)
            BooleanSpec(
                uti_relevant_antibiotics_administrations_outcome_timestamps,
                lookbehind_distances=[datetime.timedelta(days=i) for i in [30, 365]],
            ),
        ]
    }

    # Process feature specifications
    layer_spec_pairs = [
        LayerSpecPair(layer, spec)
        for layer, spec_list in feature_layers.items()
        for spec in spec_list
    ]

    logging.info("Loading specifications")
    specs = [_pair_to_spec(layer_spec) for layer_spec in layer_spec_pairs]

    logging.info("Generating feature set")
    feature_set_dir = get_uti_project_info().flattened_dataset_dir / feature_set_name

    if feature_set_dir.exists():
        while True:
            response = input(
                f"The path '{feature_set_dir}' already exists. Do you want to potentially overwrite the contents of this folder with new feature sets? (yes/no): "
            )

            if response.lower() not in ["yes", "y", "no", "n"]:
                print("Invalid response. Please enter 'yes/y' or 'no/n'.")
            if response.lower() in ["no", "n"]:
                print("Process stopped.")
                return None
            if response.lower() in ["yes", "y"]:
                print(f"Folder '{feature_set_dir}' will be overwritten.")
                break

    # Create a directory for saving feature set
    feature_set_dir.mkdir(parents=True, exist_ok=True)

    # Create feature set (read more about the package used for this process here: https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener)
    flattened_df = create_flattened_dataset(
        feature_specs=specs,
        prediction_times_frame=uti_pred_times().prediction_times,
        n_workers=None,
        step_size=datetime.timedelta(days=365),
    )

    # Save to disk
    if save:
        feature_set_path = feature_set_dir / f"{feature_set_name}.parquet"

        logging.info(f"Writing feature set to {feature_set_path}")

        flattened_df.write_parquet(feature_set_path)

    return flattened_df


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Create feature set using antibitiotics-only outcome definition and looking 48 hours ahead
    uti_generate_features(
        outcomes="antibiotics",
        lookahead_days=2,
        feature_set_name="uti_feature_set_antibiotics_outcome_definition",
    )

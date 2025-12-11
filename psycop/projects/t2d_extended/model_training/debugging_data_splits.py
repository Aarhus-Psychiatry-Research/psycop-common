from pathlib import Path

# from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.t2d_extended.model_training.populate_t2d_registry import populate_with_t2d_extended_registry

import polars as pl

import confection
from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema


def train_baseline_model(cfg_file: Path) -> float:
    """When you just want to use a file."""
    cfg = Config().from_disk(cfg_file)
    return train_baseline_model_from_cfg(cfg)


def train_baseline_model_from_cfg(cfg: confection.Config) -> float:
    """When you want to programatically change options before training, while logging the cfg."""
    cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg))
    cfg_schema.logger.log_config(cfg)
    cfg_schema.logger.warn(
        """Config is not filled, so defaults will not be logged.
                           Waiting for https://github.com/explosion/confection/issues/47 to be resolved."""
    )

    return train_baseline_model_from_schema(cfg_schema)


def train_baseline_model_from_schema(cfg: BaselineSchema) -> float:
    """For just training"""
    result = cfg.trainer.train()

    return result




def eval_stratified_split(
    cfg: PsycopConfig, training_end_date: str, evaluation_interval: tuple[str, str]
) -> float:
    outcome_col_name: str = cfg.retrieve("trainer.outcome_col_name")
    preprocessing_pipeline = cfg.rem(
        "trainer.preprocessing_pipeline.*.temporal_col_filter"
    ).retrieve("trainer.preprocessing_pipeline")

    # Setup for experiment
    cfg = (
        cfg.mut("logger.*.mlflow.experiment_name", "T2D-extended, temporal validation")
        .add(
            "logger.*.mlflow.run_name",
            f"{training_end_date}_{evaluation_interval[0]}_{evaluation_interval[1]}",
        )
        .mut("trainer.@trainers", "split_trainer_separate_preprocessing")
        .rem("trainer.outcome_col_name")
        .rem("trainer.preprocessing_pipeline")
        .rem("trainer.n_splits")
    )

    # Python dicts are ordered, but we remove the timestamp column before generating predictions.
    # To filter based on timestamp, we need to add the filter before temporal columns are removed.
    # These shenanigans are needed to insert.
    # Handle training set setup
    cfg = (
        cfg.add("trainer.training_outcome_col_name", outcome_col_name)
        .add("trainer.training_preprocessing_pipeline", preprocessing_pipeline)
        .add(
            "trainer.training_preprocessing_pipeline.*.date_filter",
            {
                "@preprocessing": "date_filter",
                "column_name": "timestamp",
                "threshold_date": training_end_date,
                "direction": "before",
            },
        )
        .add(
            "trainer.training_preprocessing_pipeline.*.temporal_col_filter",
            {"@preprocessing": "temporal_col_filter"},
        )
    )

    # Handle validation set setup
    cfg = (
        cfg.add(  # Handle validation dataset
            "trainer.validation_data", cfg.retrieve("trainer.training_data")
        )
        .add("trainer.validation_outcome_col_name", outcome_col_name)
        .add("trainer.validation_preprocessing_pipeline", preprocessing_pipeline)
        .add(
            "trainer.validation_preprocessing_pipeline.*.date_filter_start",
            {
                "@preprocessing": "date_filter",
                "column_name": "timestamp",
                "threshold_date": evaluation_interval[0],
                "direction": "after-inclusive",
            },
        )
        .add(
            "trainer.validation_preprocessing_pipeline.*.date_filter_end",
            {
                "@preprocessing": "date_filter",
                "column_name": "timestamp",
                "threshold_date": evaluation_interval[1],
                "direction": "before",
            },
        )
        .add(
            "trainer.validation_preprocessing_pipeline.*.temporal_col_filter",
            {"@preprocessing": "temporal_col_filter"},
        )
    )

    return train_baseline_model_from_cfg(cfg)


if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_t2d_extended_registry

    # Load feature set
    cfg = PsycopConfig().from_disk(
        Path(__file__).parent.parent.parent.parent / "projects" / "t2d_extended"  / "model_training" / "t2d_extended.cfg"
    )

    dataset_dirs = cfg.retrieve("trainer.training_data.paths")

    feature_set_df = pl.read_parquet(dataset_dirs[0])

    #add year column
    feature_set_df = feature_set_df.with_columns([feature_set_df['timestamp'].dt.year().alias('year')])

    #find ids that appear before 2022
    ids_before_2022 = (
    feature_set_df.filter(pl.col('year') < 2022)
        .select('dw_ek_borger')
        .unique()
        .to_series()
    )
    #find ids that appear in 2022 or after
    ids_2022_or_later = (
        feature_set_df.filter(pl.col('year') >= 2022)
        .select('dw_ek_borger')
        .unique()
    )
    #filter out IDs that were seen before 2022
    new_ids = ids_2022_or_later.filter(~pl.col('dw_ek_borger').is_in(ids_before_2022))

    #find ids from 2022 only
    ids_2022 = (
        feature_set_df.filter(pl.col('year') == 2022)
        .select('dw_ek_borger')
        .unique()
    )

    # Load eval df from 2022
    result = eval_stratified_split(
            PsycopConfig().from_disk(Path(__file__).parent / "t2d_extended.cfg"),
            training_end_date="2018-01-01",
            evaluation_interval=(f"2022-01-01", f"2022-12-31"),
        )

    eval_df = result.df.with_columns([pl.col("pred_time_uuid").apply(lambda s: s.split('-')[0]).alias("dw_ek_borger")])


    # Convert 'dw_ek_borger' to integers using map_elements
    eval_df = eval_df.with_columns([
        pl.col("dw_ek_borger").map_elements(lambda x: int(x), return_dtype=pl.Int64)
    ])



    matches = len(list(set(eval_df["dw_ek_borger"]) & set(ids_2022_or_later["dw_ek_borger"])))
    non_matches = len(list(set(ids_2022_or_later["dw_ek_borger"]) - set(eval_df["dw_ek_borger"])))

    print("n_unique ids in eval_df (2022):")
    print(len(set(eval_df["dw_ek_borger"])))
    
    print("n_unique ids in feature_set in 2022:") 
    print(len(set(ids_2022["dw_ek_borger"])))
    
    print("Dif. between n_unique ids in eval_df (2022) and feature_set in 2022:")
    print(len(set(ids_2022["dw_ek_borger"])) - len(set(eval_df["dw_ek_borger"])))

    print("Dif. between n_pred_times in eval_df (2022) and feature_set in 2022")
    print(len(ids_2022["dw_ek_borger"]) - len(eval_df["dw_ek_borger"]))

    print("hello")
from pathlib import Path
from typing import Sequence

import polars as pl
from psycop.common.feature_generation.loaders.raw.load_lab_results import hba1c
from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.projects.t2d.paper_outputs.model_permutation.modified_dataset import (
    evaluate_pipeline_with_modified_dataset,
)
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun, SplitNames
from wasabi import Printer

msg = Printer(timestamp=True)


def keep_only_hba1c_predictors(df: pl.LazyFrame, predictor_prefix: str) -> pl.LazyFrame:
    non_hba1c_pred_cols = [
        c
        for c in df.schema
        if "hba1c" not in c
        and predictor_prefix in c
        and "pred_age" not in c
        and "pred_sex" not in c
    ]
    hba1c_only_df = df.drop(non_hba1c_pred_cols)

    non_five_year_hba1c = [
        c for c in df.schema if "pred_hba1c" in c and "1825" not in c
    ]
    five_year_hba1c_only = hba1c_only_df.drop(non_five_year_hba1c)

    non_mean_hba1c = [
        c
        for c in five_year_hba1c_only.schema
        if "pred_hba1c" in c and "_mean_" not in c
    ]
    mean_five_year_hba1c_only = five_year_hba1c_only.drop(non_mean_hba1c)

    return mean_five_year_hba1c_only


def create_hba1c_only_dataset(
    run: PipelineRun,
    output_dir_path: Path,
    input_split_names: Sequence[SplitNames],
    output_split_name: str,
    recreate_dataset: bool = True,
):
    if output_dir_path.exists() and not recreate_dataset:
        msg.info("Boolean dataset has already been created, returning")
        return

    df: pl.LazyFrame = pl.concat(
        run.inputs.get_flattened_split_as_lazyframe(split)
        for split in input_split_names
    )

    hba1c_only_df = keep_only_hba1c_predictors(
        df, predictor_prefix=run.inputs.cfg.data.pred_prefix
    )

    msg.info(f"Collecting modified df with input_splits {input_split_names}")
    hba1c_only_df = hba1c_only_df.collect()

    output_dir_path.mkdir(exist_ok=True, parents=True)
    hba1c_only_df.write_parquet(output_dir_path / f"{output_split_name}.parquet")


if __name__ == "__main__":
    from psycop.projects.t2d.paper_outputs.selected_runs import (
        BEST_EVAL_PIPELINE,
    )

    evaluate_pipeline_with_modified_dataset(
        run=BEST_EVAL_PIPELINE,
        feature_modification_fn=create_hba1c_only_dataset,
        rerun_if_exists=True,
    )

    pass

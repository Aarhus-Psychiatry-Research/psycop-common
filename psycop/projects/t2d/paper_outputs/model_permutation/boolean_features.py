from collections.abc import Sequence
from pathlib import Path

import polars as pl
from wasabi import Printer

from psycop.projects.t2d.paper_outputs.model_permutation.modified_dataset import (
    FeatureModifier,
    evaluate_pipeline_with_modified_dataset,
)
from psycop.projects.t2d.utils.pipeline_objects import SplitNames, T2DPipelineRun

msg = Printer(timestamp=True)


def convert_predictors_to_boolean(
    df: pl.LazyFrame,
    predictor_prefix: str,
) -> pl.LazyFrame:
    boolean_df = df.with_columns(
        pl.when(pl.col(f"^{predictor_prefix}.*$").is_not_null())
        .then(1)
        .otherwise(0)
        .keep_name(),
    )

    return boolean_df


class CreateBooleanDataset(FeatureModifier):
    def __init__(self):
        self.name = "Boolean dataset"

    def modify_features(
        self,
        run: T2DPipelineRun,
        output_dir_path: Path,
        input_split_names: Sequence[SplitNames],
        output_split_name: str,
        recreate_dataset: bool,
    ) -> None:
        if output_dir_path.exists() and not recreate_dataset:
            msg.info("Boolean dataset has already been created, returning")
            return

        df: pl.LazyFrame = pl.concat(
            run.inputs.get_flattened_split_as_lazyframe(split)
            for split in input_split_names
        )

        boolean_df = convert_predictors_to_boolean(
            df,
            predictor_prefix=run.inputs.cfg.data.pred_prefix,
        )

        msg.info(f"Collecting boolean df with input_splits {input_split_names}")
        boolean_df = boolean_df.collect()

        output_dir_path.mkdir(exist_ok=True, parents=True)
        boolean_df.write_parquet(output_dir_path / f"{output_split_name}.parquet")


if __name__ == "__main__":
    from psycop.projects.t2d.paper_outputs.selected_runs import (
        get_best_eval_pipeline,
    )

    evaluate_pipeline_with_modified_dataset(
        run=get_best_eval_pipeline,
        feature_modifier=CreateBooleanDataset(),
    )

    pass

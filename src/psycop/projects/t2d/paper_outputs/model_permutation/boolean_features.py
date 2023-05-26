from pathlib import Path
from typing import Sequence

import polars as pl
from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.projects.t2d.paper_outputs.selected_runs import (
    BEST_EVAL_PIPELINE,
)
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun, SplitNames
from wasabi import Printer

msg = Printer(timestamp=True)


def create_boolean_dataset(
    run: PipelineRun,
    output_dir_path: Path,
    input_split_names: Sequence[SplitNames],
    output_split_name: str,
    recreate_dataset: bool = True,
):
    if output_dir_path.exists() and not recreate_dataset:
        msg.info("Boolean dataset has already been created, returning")
        return

    cfg = run.inputs.cfg

    df: pl.LazyFrame = pl.concat(
        run.inputs.get_flattened_split_as_lazyframe(split)
        for split in input_split_names
    )

    non_boolean_pred_cols = [
        c for c in df.columns if c.startswith(cfg.data.pred_prefix)
    ]

    boolean_df = df

    for col in non_boolean_pred_cols:
        boolean_df = boolean_df.with_columns(
            pl.when(pl.col(col).is_not_null())
            .then(1)
            .otherwise(0)
            .alias(f"{col}_boolean"),
        )

    boolean_df = boolean_df.drop(non_boolean_pred_cols)

    msg.info(f"Collecting boolean df with input_splits {input_split_names}")
    boolean_df = boolean_df.collect()

    output_dir_path.mkdir(exist_ok=True, parents=True)
    boolean_df.write_parquet(output_dir_path / f"{output_split_name}.parquet")


def train_model_with_boolean_dataset(
    cfg: FullConfigSchema, boolean_dataset_dir: Path
) -> float:
    cfg.data.Config.allow_mutation = True
    cfg.data.dir = str(boolean_dataset_dir)
    cfg.data.splits_for_training = ["train"]
    cfg.data.splits_for_evaluation = ["test"]

    msg.info(f"Training model for boolean evaluation from dataset at {cfg.data.dir}")
    roc_auc = train_model(cfg=cfg)
    return roc_auc


def evaluate_pipeline_with_boolean_features(run: PipelineRun):
    msg.divider(f"Evaluating {run.name} with boolean features")
    cfg: FullConfigSchema = run.inputs.cfg

    boolean_dir = run.paper_outputs.paths.estimates / "boolean"
    boolean_dataset_dir = boolean_dir / "dataset"
    auroc_md_path = boolean_dir / "boolean_auroc.md"

    if auroc_md_path.exists():
        msg.info(f"boolean AUROC already exists for {run.name}, returning")
        return

    create_boolean_dataset(
        run=run,
        output_dir_path=boolean_dataset_dir,
        input_split_names=["train", "val"],
        output_split_name="train",
    )
    create_boolean_dataset(
        run=run,
        output_dir_path=boolean_dataset_dir,
        input_split_names=["test"],
        output_split_name="test",
    )

    # Point the model at that dataset
    auroc = train_model_with_boolean_dataset(
        cfg=cfg, boolean_dataset_dir=boolean_dataset_dir
    )

    if auroc == 0.5:
        raise ValueError(
            "Returned AUROC was 0.5, indicates that try/except block was hit"
        )

    # Write AUROC
    with auroc_md_path.open("a") as f:
        f.write(str(auroc))


if __name__ == "__main__":
    evaluate_pipeline_with_boolean_features(run=BEST_EVAL_PIPELINE)

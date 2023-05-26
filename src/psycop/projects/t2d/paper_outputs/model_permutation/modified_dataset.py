from pathlib import Path
from typing import Callable

from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun
from wasabi import Printer

msg = Printer(timestamp=True)


def train_model_with_modified_dataset(
    cfg: FullConfigSchema, boolean_dataset_dir: Path
) -> float:
    cfg.data.Config.allow_mutation = True
    cfg.data.dir = str(boolean_dataset_dir)
    cfg.data.splits_for_training = ["train"]
    cfg.data.splits_for_evaluation = ["test"]

    msg.info(f"Training model from dataset at {cfg.data.dir}")
    roc_auc = train_model(cfg=cfg)
    return roc_auc


def evaluate_pipeline_with_modified_dataset(
    run: PipelineRun, feature_modification_fn: Callable
):
    modified_name = feature_modification_fn.__name__

    msg.divider(f"Evaluating {run.name} with dataset modified by {modified_name}")
    cfg: FullConfigSchema = run.inputs.cfg

    modified_dir = run.paper_outputs.paths.estimates / modified_name
    modified_dataset_dir = modified_dir / "dataset"
    auroc_md_path = modified_dir / f"{modified_name}_auroc.md"

    if auroc_md_path.exists():
        msg.info(f"{modified_name} AUROC already exists for {run.name}, returning")
        return

    feature_modification_fn(
        run=run,
        output_dir_path=modified_dataset_dir,
        input_split_names=["train", "val"],
        output_split_name="train",
    )
    feature_modification_fn(
        run=run,
        output_dir_path=modified_dataset_dir,
        input_split_names=["test"],
        output_split_name="test",
    )

    # Point the model at that dataset
    auroc = train_model_with_modified_dataset(
        cfg=cfg, boolean_dataset_dir=modified_dataset_dir
    )

    if auroc == 0.5:
        raise ValueError(
            "Returned AUROC was 0.5, indicates that try/except block was hit"
        )

    # Write AUROC
    with auroc_md_path.open("a") as f:
        f.write(str(auroc))

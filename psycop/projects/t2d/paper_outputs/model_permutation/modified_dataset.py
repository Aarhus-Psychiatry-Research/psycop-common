from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from wasabi import Printer

from psycop.common.model_training.application_modules.train_model.main import train_model
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.projects.t2d.utils.pipeline_objects import SplitNames, T2DPipelineRun

msg = Printer(timestamp=True)


class FeatureModifier(ABC):
    def __init__(self):
        self.name: str

    @abstractmethod
    def modify_features(
        self,
        run: T2DPipelineRun,
        output_dir_path: Path,
        input_split_names: Sequence[SplitNames],
        output_split_name: str,
        recreate_dataset: bool,
    ) -> None:
        pass


def train_model_with_modified_dataset(cfg: FullConfigSchema, boolean_dataset_dir: Path) -> float:
    cfg.data.Config.allow_mutation = True
    cfg.data.dir = Path(boolean_dataset_dir)
    cfg.data.splits_for_training = ["train"]
    cfg.data.splits_for_evaluation = ["test"]

    msg.info(f"Training model from dataset at {cfg.data.dir}")
    roc_auc = train_model(cfg=cfg)
    return roc_auc


def evaluate_pipeline_with_modified_dataset(
    run: T2DPipelineRun,
    feature_modifier: FeatureModifier,
    rerun_if_exists: bool = True,
    plot_fns: Sequence[Callable[[T2DPipelineRun], None]] | None = None,
):
    modified_name = feature_modifier.name

    msg.divider(f"Evaluating {run.name} with dataset modified by {modified_name}")
    cfg: FullConfigSchema = run.inputs.cfg

    modified_dir = run.paper_outputs.paths.estimates / modified_name
    modified_dataset_dir = modified_dir / "dataset"
    auroc_md_path = modified_dir / f"{modified_name}_auroc.md"

    if auroc_md_path.exists() and not rerun_if_exists:
        msg.info(f"{modified_name} AUROC already exists for {run.name}, returning")
        return

    feature_modifier.modify_features(
        run=run,
        output_dir_path=modified_dataset_dir,
        input_split_names=["train", "val"],
        output_split_name="train",
        recreate_dataset=rerun_if_exists,
    )
    feature_modifier.modify_features(
        run=run,
        output_dir_path=modified_dataset_dir,
        input_split_names=["test"],
        output_split_name="test",
        recreate_dataset=rerun_if_exists,
    )

    # Point the model at that dataset
    auroc = train_model_with_modified_dataset(cfg=cfg, boolean_dataset_dir=modified_dataset_dir)

    msg.divider(f"AUROC was {auroc}")

    if auroc == 0.5:
        raise ValueError("Returned AUROC was 0.5, indicates that try/except block was hit")

    # Write AUROC
    with auroc_md_path.open("a") as f:
        f.write(str(auroc))

    if plot_fns:
        for plot_fn in plot_fns:
            plot_fn(run)

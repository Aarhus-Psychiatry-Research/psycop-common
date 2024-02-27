from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional

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


def evaluate_pipeline_with_modified_dataset(
    run: T2DPipelineRun,
    feature_modifier: FeatureModifier,
    rerun_if_exists: bool = True,
    plot_fns: Sequence[Callable[[T2DPipelineRun], None]] | None = None,
):
    msg.divider(f"Evaluating {run.name} with dataset modified by {feature_modifier.name}")

    # Set modified attributes
    modified_run_dir = run.paper_outputs.paths.estimates / feature_modifier.name
    run.name = f"{run.name}_{feature_modifier.name}"
    run.paper_outputs.paths.estimates = run.paper_outputs.paths.estimates / feature_modifier.name
    run.pipeline_outputs.dir_path = run.pipeline_outputs.dir_path / feature_modifier.name

    auroc_md_path = modified_run_dir / f"{feature_modifier.name}_auroc.md"
    if auroc_md_path.exists() and not rerun_if_exists:
        msg.info(f"{feature_modifier.name} AUROC already exists for {run.name}, returning")
        return

    cfg: FullConfigSchema = run.inputs.cfg
    cfg.data.model_config["frozen"] = False
    cfg.data.dir = Path(modified_run_dir / "dataset")
    cfg.data.splits_for_training = ["train"]
    cfg.data.splits_for_evaluation = ["test"]

    feature_modifier.modify_features(
        run=run,
        output_dir_path=modified_run_dir / "dataset",
        input_split_names=["train", "val"],
        output_split_name="train",
        recreate_dataset=rerun_if_exists,
    )
    feature_modifier.modify_features(
        run=run,
        output_dir_path=modified_run_dir / "dataset",
        input_split_names=["test"],
        output_split_name="test",
        recreate_dataset=rerun_if_exists,
    )

    # Point the model at that dataset
    auroc = train_model(cfg=cfg, override_output_dir=run.pipeline_outputs.dir_path)

    if auroc == 0.5:
        raise ValueError("Returned AUROC was 0.5, indicates that try/except block was hit")

    # Write AUROC
    msg.divider(f"AUROC was {auroc}")
    with auroc_md_path.open("a") as f:
        f.write(str(auroc))

    if plot_fns:
        for plot_fn in plot_fns:
            plot_fn(run)

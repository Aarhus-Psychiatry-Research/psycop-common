import sys
from pathlib import Path
from typing import Any, Literal, Optional

from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.loggers.wandb import WandbLogger

from .registry import Registry


def handle_wandb_folder():
    """
    creates a folder for the debugger which otherwise causes an error
    """
    if sys.platform == "win32":
        (Path(__file__).resolve().parents[0] / "wandb" / "debug-cli.onerm").mkdir(
            exist_ok=True,
            parents=True,
        )


@Registry.loggers.register("wandb")
def create_wandb_logger(
    save_dir: Path | str,
    experiment_name: str,
    metric_prefix: str = "",
    run_name: Optional[str] = None,
    version: Optional[str] = None,
    offline: bool = False,
    id: Optional[str] = None,  # noqa: A002
    anonymous: Optional[bool] = None,
    checkpoint_name: Optional[str] = None,
) -> WandbLogger:
    handle_wandb_folder()

    return WandbLogger(
        name=run_name,
        save_dir=f"{save_dir}/{experiment_name}/{run_name}",
        version=version,
        offline=offline,
        id=id,
        anonymous=anonymous,
        project=experiment_name,
        prefix=metric_prefix,
        checkpoint_name=checkpoint_name,
    )


@Registry.loggers.register("mlflow")
def create_mlflow_logger(
    save_dir: str,
    experiment_name: str,
    run_name: Optional[str] = None,
    metric_prefix: str = "",
    tracking_uri: Optional[str] = "http://exrhel0371.it.rm.dk:5050",
    tags: Optional[dict[str, Any]] = None,
    log_model_checkpoints_to_mlflow: Literal[True, False, "all"] = False,
    artifact_location: Optional[str] = None,
    run_id: Optional[str] = None,
) -> MLFlowLogger:
    return MLFlowLogger(
        experiment_name=experiment_name,
        prefix=metric_prefix,
        run_name=run_name,
        tracking_uri=tracking_uri,
        tags=tags,
        save_dir=f"{save_dir}/{experiment_name}/{run_name}",
        log_model=log_model_checkpoints_to_mlflow,
        artifact_location=artifact_location,
        run_id=run_id,
    )

import logging
import sys
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from lightning.pytorch.loggers import Logger as plLogger
from lightning.pytorch.loggers.mlflow import MLFlowLogger as plMLFlowLogger
from lightning.pytorch.loggers.wandb import WandbLogger as plWandbLogger

from .registry import Registry

log = logging.getLogger(__name__)


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
    offline: bool,
    run_name: Optional[str] = None,
) -> plLogger:
    handle_wandb_folder()

    return plWandbLogger(
        name=run_name,
        save_dir=save_dir,
        offline=offline,
        project=experiment_name,
    )


@Registry.loggers.register("mlflow")
def create_mlflow_logger(
    save_dir: Path | str,
    experiment_name: str,
    offline: bool = False,
    run_name: Optional[str] = None,
) -> plLogger:
    if offline:
        raise NotImplementedError("MLFlow does not support offline mode")

    return plMLFlowLogger(
        save_dir=save_dir,
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri="http://exrhel0371.it.rm.dk:5050",
    )

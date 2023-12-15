import logging
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

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


class Logger(Protocol):
    def __init__(
        self,
        save_dir: Path | str,
        experiment_name: str,
        run_name: Optional[str] = None,
        offline: bool = False,
    ) -> None:
        ...

    def get_logger(
        self,
    ) -> plLogger:
        ...


@Registry.loggers.register("wandb")
class WandbCreator(Logger):
    def __init__(
        self,
        save_dir: Path | str,
        experiment_name: str,
        offline: bool,
        run_name: Optional[str] = None,
    ) -> None:
        handle_wandb_folder()

        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.offline = offline

    def get_logger(self) -> plLogger:
        return plWandbLogger(
            name=self.run_name,
            save_dir=f"{self.save_dir}/{self.experiment_name}/{self.run_name}",
            offline=self.offline,
            project=self.experiment_name,
        )


@Registry.loggers.register("mlflow")
class MLFlowCreator(Logger):
    def __init__(
        self,
        save_dir: Path | str,
        experiment_name: str,
        offline: bool = False,
        run_name: Optional[str] = None,
    ) -> None:
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.run_name = run_name
        if offline:
            raise NotImplementedError("MLFlow does not support offline mode")

    def get_logger(self) -> plLogger:
        return plMLFlowLogger(
            save_dir=f"{self.save_dir}/{self.experiment_name}/{self.run_name}",
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            tracking_uri="http://exrhel0371.it.rm.dk:5050",
        )

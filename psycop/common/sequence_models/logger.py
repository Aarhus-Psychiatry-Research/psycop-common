from pathlib import Path
from typing import Optional

from lightning.pytorch.loggers.wandb import WandbLogger

from .registry import Registry


@Registry.loggers.register("wandb")
def create_wandb_logger(
    name: Optional[str] = None,
    save_dir: Path | str = ".",
    version: Optional[str] = None,
    offline: bool = False,
    dir: Optional[Path] = None,  # noqa: A002
    id: Optional[str] = None,  # noqa: A002
    anonymous: Optional[bool] = None,
    project: Optional[str] = None,
    prefix: str = "",
    checkpoint_name: Optional[str] = None,
) -> WandbLogger:
    return WandbLogger(
        name=name,
        save_dir=save_dir,
        version=version,
        offline=offline,
        dir=dir,
        id=id,
        anonymous=anonymous,
        project=project,
        prefix=prefix,
        checkpoint_name=checkpoint_name,
    )

"""Project configuration schemas."""

from pathlib import Path

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel


class WandbSchema(PSYCOPBaseModel):
    """Configuration for weights and biases."""

    group: str
    mode: str
    entity: str


class ProjectSchema(PSYCOPBaseModel):
    """Project configuration."""

    wandb: WandbSchema
    name: str = "psycop_model_training"
    project_path: Path
    seed: int
    gpu: bool

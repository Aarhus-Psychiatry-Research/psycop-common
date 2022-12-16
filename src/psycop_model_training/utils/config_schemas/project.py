from psycop_model_training.utils.config_schemas import BaseModel, WandbSchema, WatcherSchema


class ProjectSchema(BaseModel):
    """Project configuration."""

    wandb: WandbSchema
    name: str = "psycop_model_training"
    seed: int
    watcher: WatcherSchema
    gpu: bool


class WandbSchema(BaseModel):
    """Configuration for weights and biases."""

    group: str
    mode: str
    entity: str

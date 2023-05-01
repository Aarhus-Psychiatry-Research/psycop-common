"""Create post split pipeline.""" ""
import logging

from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.preprocessing.post_split.create_pipeline import (
    create_preprocessing_pipeline,
)
from psycop_model_training.training.train_and_predict import create_model
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


def create_post_split_pipeline(cfg: FullConfigSchema) -> Pipeline:
    """Create pipeline.

    Args:
        cfg (DictConfig): Config object

    Returns:
        Pipeline
    """
    log.info("Creating post split pipeline")

    steps = []
    preprocessing_pipe = create_preprocessing_pipeline(cfg)
    if len(preprocessing_pipe.steps) != 0:
        steps.append(("preprocessing", preprocessing_pipe))

    mdl = create_model(cfg)
    steps.append(("model", mdl))
    return Pipeline(steps)

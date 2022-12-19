from sklearn.pipeline import Pipeline

from psycop_model_training.preprocessing.post_split.create_pipeline import (
    create_preprocessing_pipeline,
)
from psycop_model_training.training.train_and_eval import create_model


def create_post_split_pipeline(cfg):
    """Create pipeline.

    Args:
        cfg (DictConfig): Config object

    Returns:
        Pipeline
    """
    steps = []
    preprocessing_pipe = create_preprocessing_pipeline(cfg)
    if len(preprocessing_pipe.steps) != 0:
        steps.append(("preprocessing", preprocessing_pipe))

    mdl = create_model(cfg)
    steps.append(("model", mdl))
    return Pipeline(steps)

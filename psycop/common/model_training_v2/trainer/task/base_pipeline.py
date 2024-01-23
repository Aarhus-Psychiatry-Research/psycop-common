from typing import Protocol

from sklearn.pipeline import Pipeline


class BasePipeline(Protocol):
    sklearn_pipe: Pipeline

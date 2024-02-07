from abc import ABC

from sklearn.pipeline import Pipeline

from ...loggers.supports_logger import SupportsLoggerMixin


class BasePipeline(ABC, SupportsLoggerMixin):
    sklearn_pipe: Pipeline

from abc import ABC
from typing import Protocol

from sklearn.pipeline import Pipeline

from ...loggers.base_logger import BaselineLogger
from ...loggers.supports_logger import SupportsLoggerMixin


class BasePipeline(ABC, SupportsLoggerMixin):
    sklearn_pipe: Pipeline

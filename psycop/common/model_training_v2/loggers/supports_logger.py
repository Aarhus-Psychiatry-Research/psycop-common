from dataclasses import dataclass

from .base_logger import BaselineLogger
from .dummy_logger import DummyLogger


class SupportsLoggerMixin:
    @property
    def logger(self) -> BaselineLogger:
        if not self._logger:
            raise ValueError(
                f"No logger has been set on {type(self).__name__}. Perhaps the parent class did not pass it down?"
            )
        return self._logger

    def set_logger(self, logger: BaselineLogger) -> None:
        """Set the logger on the instance, and all attributes which support logging."""
        self._logger = logger

        # If a class without any attributes, continue
        try:
            annotations = self.__annotations__
        except AttributeError:
            return

        for attr_name in annotations:
            attr = getattr(self, attr_name)
            if isinstance(attr, SupportsLoggerMixin):
                attr.set_logger(self.logger)

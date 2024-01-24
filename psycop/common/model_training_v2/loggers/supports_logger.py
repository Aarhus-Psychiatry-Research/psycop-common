from dataclasses import dataclass

from .base_logger import BaselineLogger
from .dummy_logger import DummyLogger


@dataclass(kw_only=True)
class SupportsLoggerMixin:
    _logger: BaselineLogger | None = None
    use_dummy_logger: bool = False  # Useful in tests

    @property
    def logger(self) -> BaselineLogger:
        if self.use_dummy_logger:
            return DummyLogger()
        if not self._logger:
            raise ValueError(
                f"No logger has been set on {type(self).__name__}. Perhaps the parent class did not pass it down?"
            )
        return self._logger

    def set_logger(self, logger: BaselineLogger) -> None:
        """Set the logger on the instance, and all attributes which support logging."""
        self._logger = logger

        for attr_name in self.__annotations__:
            attr = getattr(self, attr_name)
            if isinstance(attr, SupportsLoggerMixin):
                attr.set_logger(self.logger)

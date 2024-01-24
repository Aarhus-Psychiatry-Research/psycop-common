from dataclasses import dataclass

from .base_logger import BaselineLogger


@dataclass(kw_only=True)
class SupportsLoggerMixin:
    _logger: BaselineLogger | None = None

    # XXX: Test that this works
    def __post_init__(self):
        """Set the logger on all attributes which support loggers."""
        for attr_name in self.__annotations__:
            attr = getattr(self, attr_name)
            if isinstance(attr, SupportsLoggerMixin):
                attr.set_logger(self.logger)

    @property
    def logger(self) -> BaselineLogger:
        if not self._logger:
            raise ValueError("No logger has been set.")
        return self._logger

    def set_logger(self, logger: BaselineLogger) -> None:
        self._logger = logger

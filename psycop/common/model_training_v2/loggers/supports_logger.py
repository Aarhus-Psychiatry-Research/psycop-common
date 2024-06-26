from .base_logger import BaselineLogger


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
            attributes = dir(self)
        except AttributeError:
            return

        for attr_name in attributes:
            attr = getattr(self, attr_name)
            if isinstance(attr, SupportsLoggerMixin):
                attr.set_logger(self.logger)

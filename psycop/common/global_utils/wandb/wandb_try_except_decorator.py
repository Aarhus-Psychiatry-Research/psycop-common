"""Wandb utils."""
import traceback
from typing import Any, Callable

import wandb


def wandb_alert_on_exception(func: Callable) -> Callable:  # type: ignore
    """Alerts wandb on exception."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wandb.alert(title="Run crashed", text=traceback.format_exc())
            raise e

    return wrapper

"""Wandb utils."""
import traceback

import wandb


def wandb_alert_on_exception(func):  # noqa # type: ignore
    """Alerts wandb on exception."""

    def wrapper(*args, **kwargs):  # noqa # type: ignore
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wandb.alert(title="Run crashed", text=traceback.format_exc())
            raise e

    return wrapper

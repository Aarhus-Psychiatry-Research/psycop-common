import traceback

import wandb


def wandb_alert_on_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wandb.alert(traceback.format_exc())
            raise e

    return wrapper

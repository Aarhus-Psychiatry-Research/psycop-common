"""Misc.

utility functions.
"""
import functools
import time
import traceback
from functools import wraps

import pandas as pd
import wandb
from wasabi import Printer


def print_df_dimensions_diff(func):  # noqa
    """Print the difference in rows between the input and output dataframes."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa
        msg = Printer(timestamp=True)

        start_time = time.time()

        base_msg = f"{func.__name__}"

        arg_dfs = [arg for arg in args if isinstance(arg, pd.DataFrame)]
        kwargs_dfs = [arg for arg in kwargs.values() if isinstance(arg, pd.DataFrame)]
        potential_dfs = arg_dfs + kwargs_dfs

        if len(potential_dfs) > 1:
            raise ValueError("More than one DataFrame found in args or kwargs")

        df = potential_dfs[0]

        diff_msg = ""

        for dim in ("rows", "columns"):
            dim_int = 0 if dim == "rows" else 1

            n_in_dim_before_func = df.shape[dim_int]

            result = func(*args, **kwargs)

            diff = n_in_dim_before_func - result.shape[dim_int]

            if diff != 0:
                percent_diff = round(
                    (diff) / n_in_dim_before_func * 100,
                    2,
                )

                diff_msg += f"Dropped {diff} ({percent_diff}%) {dim}, started with {n_in_dim_before_func} {dim}. | "

        end_time = time.time()
        duration_str = f"{int(end_time - start_time)} s"

        msg.info(f"{duration_str} | {base_msg} | {diff_msg}")

        return result  # type: ignore

    return wrapper


def wandb_alert_on_exception_return_terrible_auc(func):  # noqa
    """Alerts wandb on exception."""

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa
        try:
            return func(*args, **kwargs)
        except Exception:  # pylint: disable=broad-except, unused-variable
            if wandb.run is not None:
                wandb.alert(title="Run crashed", text=traceback.format_exc())
            return 0.5

    return wrapper

"""Utility functions for the application modules."""
import functools
import logging
from typing import Callable

import pandas as pd


def print_df_dimensions_diff(
    func: Callable,
    print_when_starting: bool = False,
    print_when_no_diff=False,
):
    """Print the difference in rows between the input and output dataframes."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log = logging.getLogger(__name__)

        if print_when_starting:
            log.info(f"{func.__name__}: Starting")

        arg_dfs = [arg for arg in args if isinstance(arg, pd.DataFrame)]
        kwargs_dfs = [arg for arg in kwargs.values() if isinstance(arg, pd.DataFrame)]
        potential_dfs = arg_dfs + kwargs_dfs

        if len(potential_dfs) > 1:
            raise ValueError("More than one DataFrame found in args or kwargs")

        df = potential_dfs[0]

        for dim in ("rows", "columns"):
            dim_int = 0 if dim == "rows" else 1

            n_in_dim_before_func = df.shape[dim_int]

            if print_when_no_diff:
                log.info(
                    f"{func.__name__}: {n_in_dim_before_func} {dim} before function",
                )

            result = func(*args, **kwargs)

            diff = n_in_dim_before_func - result.shape[dim_int]

            if diff != 0:
                percent_diff = round(
                    (n_in_dim_before_func - result.shape[dim_int])
                    / n_in_dim_before_func
                    * 100,
                    2,
                )

                log.info(f"{func.__name__}: Dropped {diff} ({percent_diff}%) {dim}")
            else:
                if print_when_no_diff:
                    log.info(f"{func.__name__}: No {dim} dropped")

        return result

    return wrapper

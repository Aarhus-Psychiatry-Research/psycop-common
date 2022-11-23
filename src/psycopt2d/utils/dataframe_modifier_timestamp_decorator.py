"""Print the difference in rows between the input and output dataframes."""
import functools

import pandas as pd
from wasabi import Printer


def print_diff_rows(func):
    """Print the difference in rows between the input and output dataframes."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        msg = Printer(timestamp=True)

        arg_dfs = [arg for arg in args if isinstance(arg, pd.DataFrame)]
        kwargs_dfs = [arg for arg in kwargs.values() if isinstance(arg, pd.DataFrame)]
        potential_dfs = arg_dfs + kwargs_dfs

        if len(potential_dfs) > 1:
            raise ValueError("More than one DataFrame found in args or kwargs")

        df = potential_dfs[0]

        n_rows_before_func = df.shape[0]

        msg.info(f"{func.__name__}: {n_rows_before_func} rows before function")

        result = func(*args, **kwargs)

        percent_diff = round(
            (n_rows_before_func - result.shape[0]) / n_rows_before_func,
            2,
        )

        msg.info(f"{func.__name__}: Dropped {percent_diff}% of rows")

        return result

    return wrapper

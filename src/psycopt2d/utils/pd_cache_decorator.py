"""Pandas cache decorator."""
import pathlib
from functools import wraps

import pandas as pd


def cache_pandas_result(cache_dir: pathlib.Path, hard_reset: bool = False):
    """This decorator caches a pandas.DataFrame returning function.

    It saves the pandas.DataFrame in a parquet file in the cache_dir named a hash of its name, args and kwargs.
    It uses the following naming scheme for the caching files:
        cache_dir / function_name + '.trc.pqt'
    Parameters:
    cache_dir: a pathlib.Path object
    hard_reset: bool
    """

    def build_caching_function(func):
        @wraps(func)
        def cache_function(*args, **kwargs):
            if not isinstance(cache_dir, pathlib.Path):
                raise TypeError("cache_dir should be a pathlib.Path object")

            # Hash args to a string
            hash_str = str(hash((args, tuple(kwargs), func.__name__)))

            cache_file_path = cache_dir / (f"{hash_str}.trc.pqt")

            if not cache_dir.exists():
                print("Creating cache dir")
                cache_dir.mkdir(exist_ok=True, parents=True)

            if hard_reset or (not cache_file_path.exists()):
                result = func(*args, **kwargs)

                if not isinstance(result, pd.DataFrame):
                    raise TypeError(
                        f"The result of computing {func.__name__} is not a DataFrame",
                    )

                result.to_parquet(cache_file_path)
                return result

            result = pd.read_parquet(cache_file_path)
            return result

        return cache_function

    return build_caching_function

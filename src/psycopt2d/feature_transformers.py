"""Contains custom transformers for data preprocessing."""
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ConvertToBoolean(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns_to_include: Optional[List[str]] = None,
        columns_to_skip: Optional[List[str]] = ["age_in_years", "sex_female"],
        ignore_dtypes: set = {"datetime64[ns]"},
    ) -> None:
        """Convert variables to boolean, used for checking whether a column has
        a value.

        Args:
            columns_to_include (List[str], optional): Columns to convert to boolean.
                Acts as a whitelist, skipping all columns not in the list.
            columns_to_skip (List[str], optional): Columns to not convert to boolean.
                Acts as a blacklist.
                Defaults to ["age_in_years", "male"].
                Default to None in which case all columns are included.
            ignore_dtypes (set, optional): Skip columns with these data types. Defaults
                to {"datetime64[ns]"}.
        """
        self.columns_to_skip = columns_to_skip
        self.columns_to_include = columns_to_include
        self.ignore_dtypes = ignore_dtypes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        columns = X.columns

        if self.columns_to_include:
            columns = [c for c in columns if c in self.columns_to_include]
        cols_to_round = [
            c
            for c in columns
            if (X[c].dtype not in self.ignore_dtype) or c in self.columns_to_skip
        ]

        for col in cols_to_round:
            X[col] = X[col].map(lambda x: 1 if not pd.isnull(x) else np.NaN)
        return X, y


class DateTimeConverter(BaseEstimator, TransformerMixin):
    valid_types = {"ordinal"}
    datetime_dtypes = {"datetime64[ns]"}

    def __init__(self, convert_to="ordinal"):
        """Convert datetimes to integers.

        Args:
            convert_to (str, optional): What should the datetime be converted to?
                Defaults to "ordinal".
                Options include:
                    - "ordinal": In which case it uses the `datetime.toordinal` to
                      convert to ordinal.
        """
        if convert_to not in self.valid_types:
            raise ValueError(
                f"{convert_to} is not a valid type, valid types include "
                + f"'{self.valid_types}'",
            )

        self.convert_to = convert_to

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # extract datatime columns
        dt_columns = [c for c in X.columns if X[c].dtypes in self.datetime_dtypes]

        # convert datetime columns
        for colname in dt_columns:
            X[colname] = X[colname].map(datetime.toordinal)

        return X, y

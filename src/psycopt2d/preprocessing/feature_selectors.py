"""Custom feature selectors."""
from sklearn.base import BaseEstimator, TransformerMixin


class DropDateTimeColumns(BaseEstimator, TransformerMixin):
    """Convert all cells with a value to True, otherwise false."""

    def __init__(
        self,
        pred_prefix: str,
        drop_dtypes: tuple = ("datetime64[ns]", "<M8[ns]"),
    ) -> None:
        """
        Args:
            drop_dtypes (set, optional): Drop columns with these data types.
        """
        self.drop_dypes = drop_dtypes
        self.pred_prefix = pred_prefix

    def fit(self, _, y=None):  # pylint: disable=unused-argument
        """Fit the transformer."""
        return self

    def transform(self, X, y=None):  # pylint: disable=unused-argument
        """Transform the data."""
        columns_to_drop = [c for c in X.columns if X[c].dtype in self.drop_dypes]
        columns_to_drop = [c for c in columns_to_drop if c.startswith(self.pred_prefix)]

        return X[[c for c in X.columns if c not in columns_to_drop]]

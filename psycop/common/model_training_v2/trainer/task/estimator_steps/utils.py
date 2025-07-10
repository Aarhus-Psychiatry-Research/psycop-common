from sklearn.base import BaseEstimator, TransformerMixin


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):  # type: ignore # noqa
        return self

    def transform(self, input_array, y=None):  # type: ignore # noqa
        return input_array

# first line: 19
@shared_cache.cache()
def cached_folds(
    n_splits: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,  # type: ignore
) -> list[tuple[ndarray[Any, Any], ndarray[Any, Any]]]:
    return list(StratifiedGroupKFold(n_splits=n_splits).split(X=X, y=y, groups=groups))

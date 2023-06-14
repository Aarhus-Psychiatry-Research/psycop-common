import pickle
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
import polars as pl
import shap
from care_ml.model_evaluation.config import EVAL_RUN
from care_ml.model_evaluation.utils.feature_name_to_readable import (
    feature_name_to_readable,
)
from care_ml.utils.best_runs import Run
from joblib import Memory
from psycop.common.global_utils.cache import mem
from sklearn.pipeline import Pipeline

mem = Memory(location=".", verbose=0)  # noqa: F811


def generate_shap_df_for_predictor_col(
    colname: str,
    X: pd.DataFrame,
    shap_values: list[float],
) -> pd.DataFrame:
    colname_index = X.columns.get_loc(colname)

    df = pd.DataFrame(
        {
            "feature_name": colname,
            "feature_value": X[colname],
            "pred_time_index": list(range(0, len(X))),
            "shap_value": shap_values[:, colname_index],  # type: ignore
        },
    )

    return df


@mem.cache
def get_long_shap_df(X: pd.DataFrame, shap_values: list[float]) -> pd.DataFrame:
    predictor_cols = X.columns
    dfs = []

    for c in predictor_cols:
        dfs.append(
            generate_shap_df_for_predictor_col(
                colname=c,
                X=X,
                shap_values=shap_values,
            ),
        )

    return pd.concat(dfs, axis=0)


@dataclass
class ShapBundle:
    shap_values: list[float]
    X: pd.DataFrame

    def get_long_shap_df(self) -> pd.DataFrame:
        """Returns a long dataframe with columns:
        * feature_name (e.g. "age")
        * feature_value (e.g. 31)
        * pred_time_index (e.g. "010573-2020-01-01")
        * shap_value (e.g. 0.1)
        Each row represents an observation of a feature at a prediction time.
        """
        return get_long_shap_df(X=self.X, shap_values=self.shap_values)  # type: ignore


def generate_shap_values_from_pipe(
    features: pl.LazyFrame,
    outcome: pl.LazyFrame,
    pipeline: Pipeline,
) -> list[float]:
    numerical_predictors = []

    for c in features.schema:
        if features.schema[c] == pl.Float64 and c.startswith("pred_"):
            numerical_predictors.append(c)

    features = features.with_columns(pl.col(numerical_predictors).round(1).keep_name())

    features_df = features.collect().to_pandas()
    outcome_df = outcome.collect().to_pandas()

    model = pipeline["model"]  # type: ignore
    explainer = shap.TreeExplainer(model)  # type: ignore
    shap_values = explainer.shap_values(features_df, y=outcome_df)
    return shap_values


@mem.cache
def get_shap_bundle_for_best_run(
    run: Run = EVAL_RUN,
    n_rows: Optional[int] = 10_000,
    cache_ver: float = 0.1,
    split_name: Literal["train", "val", "test"] = "val",
) -> ShapBundle:
    print(f"Generating shap values for {run.name}, with cache version {cache_ver}")

    flattened_ds: pl.DataFrame = (
        pl.concat(
            run.get_flattened_split_as_lazyframe(split=split) for split in [split_name]  # type: ignore
        )
        .collect()  # type: ignore
        .sample(n=n_rows, with_replacement=True)
    )

    if n_rows:
        flattened_ds = flattened_ds.sample(n=n_rows, with_replacement=True)

    cfg = run.cfg
    predictor_cols = [
        c for c in flattened_ds.columns if c.startswith(cfg.data.pred_prefix)
    ]
    outcome_cols = [
        c
        for c in flattened_ds.columns
        if c.startswith(cfg.data.outc_prefix)
        and str(cfg.preprocessing.pre_split.min_lookahead_days) in c
    ]

    pipe = run.pipe

    shap_values = generate_shap_values_from_pipe(
        features=flattened_ds.lazy().select(predictor_cols),
        outcome=flattened_ds.lazy().select(outcome_cols),
        pipeline=pipe,  # type: ignore
    )

    return ShapBundle(
        shap_values=shap_values,
        X=flattened_ds.select(predictor_cols).to_pandas(),
    )


def get_top_i_features_by_mean_abs_shap(
    shap_long_df: pl.DataFrame,
    i: int,
) -> pl.DataFrame:
    feature_shap_agg = shap_long_df.groupby("feature_name").agg(
        shap_mean=pl.col("shap_value").abs().mean(),
    )

    feature_shap_agg_with_ranks = feature_shap_agg.with_columns(
        shap_mean_rank=pl.col("shap_mean")
        .rank(method="average", descending=True)
        .cast(pl.Int32),
    )

    selected_features = feature_shap_agg_with_ranks.filter(
        i >= pl.col("shap_mean_rank"),
    )

    return selected_features.join(shap_long_df, on="feature_name", how="left").drop(
        "shap_mean",
    )


if __name__ == "__main__":
    shap_bundle = get_shap_bundle_for_best_run(
        run=EVAL_RUN,
        n_rows=1_000,
        cache_ver=0.1,
    )

    long_shap_df = shap_bundle.get_long_shap_df()  # type: ignore

    pass


@mem.cache
def generate_shap_values(
    features: pd.DataFrame,
    outcome: pd.DataFrame,
    pipeline: Pipeline,
) -> bytes:
    for feature in features.columns:
        if len(features[feature].unique()) > 100:
            features[feature] = features[feature].round(1)

    # rename features
    features.columns = [
        feature_name_to_readable(col, warning=False) for col in features.columns
    ]

    model = pipeline["model"]
    explainer = shap.TreeExplainer(model)  # type: ignore
    shap_values = explainer(features, y=outcome)

    return pickle.dumps(shap_values)

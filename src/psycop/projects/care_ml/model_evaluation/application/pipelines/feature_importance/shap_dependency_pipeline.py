"""Scatter plots for top i features, as defined by largest absolute mean SHAP"""

from typing import Literal, Optional

import polars as pl
from care_ml.model_evaluation.config import (
    EVAL_RUN,
    FIGURES_PATH,
    TABLES_PATH,
    TEXT_EVAL_RUN,
    TEXT_FIGURES_PATH,
    TEXT_TABLES_PATH,
)
from care_ml.model_evaluation.figures.feature_importance.shap.get_shap_values import (
    get_shap_bundle_for_best_run,
)
from care_ml.model_evaluation.figures.feature_importance.shap.shap_plots import (
    save_plots_for_top_i_shap_by_mean_abs,
)
from care_ml.model_evaluation.figures.feature_importance.shap.shap_table import (
    get_top_i_shap_values_for_printing,
)


def shap_dependency_pipeline(
    model: Literal["baseline", "text"],
    top_n_shap_table: Optional[int] = 100,
    top_n_shap_scatter: Optional[int] = 20,
):
    if model == "baseline":
        run, f_path, t_path = EVAL_RUN, FIGURES_PATH, TABLES_PATH
    elif model == "text":
        run, f_path, t_path = TEXT_EVAL_RUN, TEXT_FIGURES_PATH, TEXT_TABLES_PATH
    else:
        raise ValueError(f"model is {model}, but must be 'baseline' or 'text'")

    long_shap_df = get_shap_bundle_for_best_run(
        run=run,
        n_rows=100_000,
        cache_ver=0.01,
    ).get_long_shap_df()  # type: ignore

    if top_n_shap_table:
        # write table with shap values for top i features (largest abs SHAP mean)
        table_df = get_top_i_shap_values_for_printing(
            shap_long_df=pl.from_pandas(long_shap_df),
            i=top_n_shap_table,
        )
        table_df.write_csv(file=t_path / "shap_table.csv")

    if top_n_shap_scatter:
        # create and save scatter plots for top i features (largest abs SHAP mean)
        plotting_df = pl.from_pandas(long_shap_df)

        shap_figures_path = f_path / "shap"
        shap_figures_path.mkdir(exist_ok=True, parents=True)

        save_plots_for_top_i_shap_by_mean_abs(
            shap_long_df=plotting_df,
            i=top_n_shap_scatter,
            save_dir=shap_figures_path,
            model=model,
        )


if __name__ == "__main__":
    shap_dependency_pipeline(model="baseline", top_n_shap_table=None)
    shap_dependency_pipeline(model="text", top_n_shap_table=None)

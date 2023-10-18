from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline

if __name__ == "__main__":
    from psycop.projects.t2d.paper_outputs.model_description.feature_importance.shap.get_shap_values import (
        get_shap_bundle_for_best_run,
    )

    long_shap_df = get_shap_bundle_for_best_run(
        n_rows=100_000,
        cache_ver=0.01,
    ).get_long_shap_df()  # type: ignore

    import polars as pl

    from psycop.projects.t2d.paper_outputs.model_description.feature_importance.shap.shap_table import (
        get_top_i_shap_values_for_printing,
    )

    table_df = get_top_i_shap_values_for_printing(
        shap_long_df=pl.from_pandas(long_shap_df),
        i=100,
    )

    plotting_df = pl.from_pandas(long_shap_df)

    shap_figures_path = (
        get_best_eval_pipeline().paper_outputs.paths.figures / "shap_plot.png"
    )
    shap_figures_path.mkdir(exist_ok=True, parents=True)

    from psycop.projects.t2d.paper_outputs.model_description.feature_importance.shap.plot_shap import (
        save_plots_for_top_i_shap_by_mean_abs,
    )

    save_plots_for_top_i_shap_by_mean_abs(
        shap_long_df=plotting_df,
        i=10,
        save_dir=shap_figures_path,
    )

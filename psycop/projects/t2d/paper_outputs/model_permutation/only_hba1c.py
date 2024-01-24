from collections.abc import Sequence
from pathlib import Path

import polars as pl
from wasabi import Printer

from psycop.projects.t2d.paper_outputs.model_description.performance.main_performance_figure import (
    t2d_create_main_performance_figure,
)
from psycop.projects.t2d.paper_outputs.model_permutation.modified_dataset import (
    FeatureModifier,
    evaluate_pipeline_with_modified_dataset,
)
from psycop.projects.t2d.utils.pipeline_objects import SplitNames, T2DPipelineRun

msg = Printer(timestamp=True)


class Hba1cOnly(FeatureModifier):
    def __init__(self, lookbehind: str, aggregation_method: str, name: str = "hba1c_only"):
        self.name = name
        self.lookahead = lookbehind
        self.aggregation_method = aggregation_method

    def modify_features(
        self,
        run: T2DPipelineRun,
        output_dir_path: Path,
        input_split_names: Sequence[SplitNames],
        output_split_name: str,
        recreate_dataset: bool,
    ) -> None:
        if output_dir_path.exists() and not recreate_dataset:
            msg.info("Boolean dataset has already been created, returning")
            return

        df: pl.LazyFrame = pl.concat(
            run.inputs.get_flattened_split_as_lazyframe(split) for split in input_split_names
        )

        hba1c_only_df = self._keep_only_hba1c_predictors(
            df, predictor_prefix=run.inputs.cfg.data.pred_prefix
        )

        msg.info(f"Collecting modified df with input_splits {input_split_names}")
        hba1c_only_df = hba1c_only_df.collect()

        output_dir_path.mkdir(exist_ok=True, parents=True)
        hba1c_only_df.write_parquet(output_dir_path / f"{output_split_name}.parquet")

    def _keep_only_hba1c_predictors(self, df: pl.LazyFrame, predictor_prefix: str) -> pl.LazyFrame:
        non_hba1c_pred_cols = [
            c
            for c in df.schema
            if "hba1c" not in c
            and predictor_prefix in c
            and "pred_age" not in c
            and "pred_sex" not in c
        ]
        hba1c_only_df = df.drop(non_hba1c_pred_cols)

        non_lookahead_hba1c = [
            c for c in df.schema if "pred_hba1c" in c and self.lookahead not in c
        ]
        lookahead_hba1c_only = hba1c_only_df.drop(non_lookahead_hba1c)

        non_aggregation_method_hba1c = [
            c
            for c in lookahead_hba1c_only.schema
            if "pred_hba1c" in c and f"_{self.aggregation_method}_" not in c
        ]
        mean_five_year_hba1c_only = lookahead_hba1c_only.drop(non_aggregation_method_hba1c)

        return mean_five_year_hba1c_only


if __name__ == "__main__":
    from copy import copy

    from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline

    run = copy(get_best_eval_pipeline())
    run.name = "xgboost_hba1c_only"
    default_xgboost_params = True

    if default_xgboost_params:
        msg.divider("Training with default xgboost params")
        cfg = run.inputs.cfg

        # Set XGBoost to default hyperparameters
        cfg.model.model_config["frozen"] = False
        cfg.model.args = {
            "n_estimators": 100,
            "alpha": 0,
            "lambda": 1,
            "max_depth": 6,
            "learning_rate": 0.3,
            "gamma": 0,
            "grow_policy": "depthwise",
        }

    evaluate_pipeline_with_modified_dataset(
        run=run,
        feature_modifier=Hba1cOnly(lookbehind="730", aggregation_method="mean"),
        rerun_if_exists=True,
        plot_fns=[t2d_create_main_performance_figure],
    )

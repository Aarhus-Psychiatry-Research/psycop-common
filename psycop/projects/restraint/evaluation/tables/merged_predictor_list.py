from typing import Any, Callable
import pandas as pd
import polars as pl
from pathlib import Path
import re
import numpy as np

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.config_utils import resolve_and_fill_config
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.loggers.terminal_logger import TerminalLogger
from psycop.projects.restraint.evaluation.tables.restraint_predictor_describer import ParsedPredictorColumn, tsflattener_v2_column_is_static, _get_match_group
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR

def get_filtered_prediction_times(cfg: dict[str, Any]) -> pl.DataFrame:
        data = cfg["trainer"].training_data.load()

        preprocessing_pipeline = cfg["trainer"].training_preprocessing_pipeline
        preprocessed_all_splits: pl.DataFrame = pl.from_pandas(preprocessing_pipeline.apply(data))

        return preprocessed_all_splits

def parse_predictor_column_name(
    col_name: str,
    is_static: Callable[[str], bool] = tsflattener_v2_column_is_static,
    feature_name_regex: str = r"[a-zA-Z]+_(.*?)_(?=fallback|within)",
    fallback_regex: str = r"_fallback_(\d+|nan)",
    time_interval_start_regex: str = r"_within_([0-9]+)",
    resolve_multiple_strategy_regex: str = r"([a-z]+)_fallback",
) -> ParsedPredictorColumn:
    if is_static(col_name):
        return ParsedPredictorColumn(
            col_name=col_name,
            feature_name=re.search(r"[a-zA-Z]+_(.+)", col_name).group(1),  # type: ignore
            fallback="0",
            time_interval_start="nan",
            resolve_multiple_strategy="nan",
            is_static=True,
        )
    
    feature_name = _get_match_group(feature_name_regex, col_name)
    fallback = _get_match_group(fallback_regex, col_name)
    time_interval_start = _get_match_group(time_interval_start_regex, col_name)
    resolve_multiple_strategy = _get_match_group(resolve_multiple_strategy_regex, col_name)

    return ParsedPredictorColumn(
        col_name=col_name,
        feature_name=feature_name,
        fallback=fallback,
        time_interval_start=time_interval_start,
        resolve_multiple_strategy=resolve_multiple_strategy,
        is_static=is_static(col_name),
    )

def generate_feature_description_df_minimal(
    cols: list[str],
    column_name_parser: Callable[[str], ParsedPredictorColumn] = parse_predictor_column_name,
) -> pl.DataFrame:
    errors = []
    parsed_cols: list[ParsedPredictorColumn] = []
    for col in cols:
        try:
            parsed_cols.append(column_name_parser(col))
        except ValueError as e:
            errors.append(e)
            continue

    if errors:
        print(errors)
        raise ValueError(errors)

    feature_rows = []
    for parsed_col in parsed_cols:
        if "tfidf" in parsed_col.feature_name:
            vocab = pl.read_parquet(
                OVARTACI_SHARED_DIR
                / "text_models"
                / "vocabulary_lists"
                / "vocab_tfidf_psycop_train_all_sfis_preprocessed_sfi_type_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.parquet"
            )

            tfidf_idx = re.search(r"(\d+)", parsed_col.feature_name).group(0)  # type: ignore
            tfidf_word = vocab.filter(pl.col("Index") == int(tfidf_idx))["Word"][0]
            parsed_col.feature_name = re.sub(tfidf_idx, tfidf_word, parsed_col.feature_name)

        feature_description = {
            "Feature name": parsed_col.feature_name,
            "Lookbehind period": parsed_col.time_interval_start
            if parsed_col.is_static
            else parsed_col.time_interval_start,
            "Resolve multiple": parsed_col.resolve_multiple_strategy,
            "Fallback strategy": parsed_col.fallback,
            "Static": parsed_col.is_static,
        }
        feature_rows.append(feature_description)

    feature_description_df = pl.DataFrame(feature_rows)
    return feature_description_df.sort("Feature name")

def merge_predictor_lists(experiment_name: str, model_name: str, pred_list: pd.DataFrame) -> pd.DataFrame:
    cfg_path = Path(f"E:/shared_resources/restraint/eval_runs/{experiment_name}/config.cfg")
    
    resolved_cfg = resolve_and_fill_config(cfg_path, fill_cfg_with_defaults=True)
    resolved_cfg["trainer"].training_preprocessing_pipeline._logger = TerminalLogger()

    input_df = get_filtered_prediction_times(resolved_cfg).drop([resolved_cfg["trainer"].uuid_col_name, resolved_cfg["trainer"].group_col_name, resolved_cfg["trainer"].training_outcome_col_name])

    input_features = np.array(input_df.columns)

    run = MlflowClientWrapper().get_best_run_from_experiment(experiment_name, metric="all_oof_BinaryAUROC")
    if "Minimal" not in model_name:
        mask=run.sklearn_pipeline()["feature_selection"].get_support()
        selected_features=input_features[mask]
    else:
        selected_features = input_features

    feat_list = [col for col in selected_features]

    feat_df = generate_feature_description_df_minimal(feat_list).to_pandas()

    feat_df["Lookbehind period"][feat_df["Lookbehind period"] == "nan"] = np.nan
    feat_df["Lookbehind period"] = pd.to_numeric(feat_df["Lookbehind period"])

    feat_df["Fallback strategy"][feat_df["Fallback strategy"] == "nan"] = np.nan
    feat_df["Fallback strategy"] = pd.to_numeric(feat_df["Fallback strategy"])

    merged=pred_list.merge(feat_df, how="outer", on=["Feature name", "Lookbehind period", "Resolve multiple", "Fallback strategy", "Static"], indicator=True)

    merged[model_name] = merged["_merge"].replace({"left_only": " ", "both": "X"})

    return merged.drop(columns="_merge")


if __name__ == "__main__":
    pred_list = pd.read_csv("E:/shared_resources/restraint/eval_runs/feature_description.csv")
    pred_list["Resolve multiple"] = pred_list["Resolve multiple"].replace({np.nan: "nan"})

    populate_baseline_registry()

    merged_dfs = []

    for experiment, model in zip(["restraint_mechanical_tuning_v2_best_run_evaluated_on_test", "restraint_all_tuning_v2_best_run_evaluated_on_test", "restraint_mechanical_tuning_minimal_v2_best_run_evaluated_on_test"], ["Mechanical model", "Composite model", "Minimal models"]):   
        merged_dfs.append(merge_predictor_lists(experiment_name=experiment, model_name=model, pred_list=pred_list))
    
    complete_df = merged_dfs[0].merge(merged_dfs[1], how="inner", on=['Feature name', 'Lookbehind period', 'Resolve multiple', 'Fallback strategy', 'Static', 'Mean', 'N. unique','Proportion using fallback']).merge(merged_dfs[2], how="inner", on=['Feature name', 'Lookbehind period', 'Resolve multiple', 'Fallback strategy', 'Static', 'Mean', 'N. unique','Proportion using fallback'])

    complete_df.to_csv("E:/shared_resources/restraint/eval_runs/feature_description_merged.csv")
import polars as pl

from psycop.common.model_training_v2.trainer.preprocessing.steps.column_filters import (
    FilterColumnsWithinSubset,
)

if __name__ == "__main__":
    filters = [
        FilterColumnsWithinSubset(subset_rule="eval.+", keep_matching="keep_none"),
        FilterColumnsWithinSubset(subset_rule="timestamp_.+", keep_matching="keep_none"),
        FilterColumnsWithinSubset(subset_rule="pred_.+", keep_matching=".+hba1c.+|.+age.+|.+sex.+"),
        FilterColumnsWithinSubset(subset_rule="pred.+within.+", keep_matching=".+730.+"),
        FilterColumnsWithinSubset(subset_rule="pred.+within.+", keep_matching=".+mean.+"),
        FilterColumnsWithinSubset(subset_rule="outc.+", keep_matching=".+1825.+"),
    ]

    old_paths = [
        "E:/shared_resources/t2d/feature_sets/psycop_t2d_adminmanber_features_2023_04_27_14_25/psycop_t2d_adminmanber_features_2023_04_27_14_25_train.parquet",
        "E:/shared_resources/t2d/feature_sets/psycop_t2d_adminmanber_features_2023_04_27_14_25/psycop_t2d_adminmanber_features_2023_04_27_14_25_val.parquet",
        "E:/shared_resources/t2d/feature_sets/psycop_t2d_adminmanber_features_2023_04_27_14_25/psycop_t2d_adminmanber_features_2023_04_27_14_25_test.parquet",
    ]
    old = pl.concat(pl.scan_parquet(p) for p in old_paths)

    old_f = old
    for f in filters:
        old_f = f.apply(old_f)

    old2new_cols = {
        "prediction_time_uuid": "prediction_time_uuid",
        "outc_first_diabetes_lab_result_within_1825_days_max_fallback_0_dichotomous": "outc_t2d_value_within_0_to_1825_days_max_fallback_0",
        "pred_sex_female": "pred_sex_female_fallback_0",
        "pred_age_in_years": "pred_age_days_fallback_0",
        "pred_hba1c_within_730_days_mean_fallback_nan": "pred_layer_1_hba1c_within_0_to_730_days_mean_fallback_nan",
    }

    new = (
        pl.scan_parquet(
            "E:/shared_resources/t2d_extended/flattened_datasets/t2d_extended_feature_set/t2d_extended_feature_set.parquet"
        )
        .select(old2new_cols.values())
        .collect()
        .sort("prediction_time_uuid")
    )

    ## Make as similar as possible
    old_f_renamed = (
        old_f.rename(old2new_cols)
        .select(old2new_cols.values())
        .collect()
        .sort("prediction_time_uuid")
        .with_columns(pl.col("pred_age_days_fallback_0") * 365.25)
    )

    ## Match things that did not before
    new_matched = new.with_columns(
        pl.col("prediction_time_uuid")
        .str.replace(r"\.000000000", "")
        .str.replace_all(":", "-")
        .str.replace_all(" ", "-")
    )

    from datacompy import PolarsCompare

    comparison = PolarsCompare(
        old_f_renamed, new_matched, join_columns=["prediction_time_uuid"], rel_tol=0.00
    )
    print(comparison.report())

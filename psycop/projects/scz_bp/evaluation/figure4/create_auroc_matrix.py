from pathlib import Path

import pandas as pd
import polars as pl
from confection import Config
from sklearn.metrics import roc_auc_score

from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.scz_bp.dataset_description.scz_bp_table_one import SczBpTableOne
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_best_run_in_experiment,
)


def scz_bp_merge_eval_dfs(eval_ds: EvalDataset) -> pl.DataFrame:
    cfg = Config().from_disk(Path(__file__).parent / "eval_config.cfg")
    meta_df = SczBpTableOne(cfg).get_filtered_prediction_times()
    meta_df_filtered = meta_df.with_columns(
        pl.concat_str([pl.col("dw_ek_borger"), pl.col("timestamp")], separator="-").alias(
            "pred_time_uuids"
        )
    )

    meta_df_filtered.select(
        "pred_time_uuid",
        "meta_scz_diagnosis_within_0_to_1825_days_max_fallback_0",
        "meta_bp_diagnosis_within_0_to_1825_days_max_fallback_0",
        "outc_first_scz_or_bp_within_0_to_1825_days_max_fallback_0",
    )

    eval_df = eval_ds.to_polars().with_columns(
        pl.concat_str([pl.col("ids"), pl.col("pred_timestamps")], separator="-").alias(
            "pred_time_uuid"
        )
    )

    joined_df = eval_df.join(meta_df_filtered, on="pred_time_uuids", how="left", validate="1:1")

    return joined_df.rename(
        {
            "meta_scz_diagnosis_within_0_to_1825_days_max_fallback_0": "scz_diagnosis",
            "meta_bp_diagnosis_within_0_to_1825_days_max_fallback_0": "bp_diagnosis",
            "outc_first_scz_or_bp_within_0_to_1825_days_max_fallback_0": "first_diagnosis",
        }
    )


def scz_bp_auroc_by_outcome(model2validation_mapping: list[tuple[str, str]]) -> pd.DataFrame:
    auroc_df = pd.DataFrame(columns=["scz_diagnosis", "first_diagnosis", "bp_diagnosis"])
    for model_name, validation_outcome in model2validation_mapping:
        eval_df = scz_bp_merge_eval_dfs(
            scz_bp_get_eval_ds_from_best_run_in_experiment(experiment_name=model_name)
        )
        eval_df = eval_df.filter(pl.col(validation_outcome).is_not_null()) #lasse 

        auroc_df.loc[model_name, validation_outcome] = roc_auc_score(  # type: ignore
            y_true=eval_df[validation_outcome], y_score=eval_df["y_hat_probs"]
        )

    return auroc_df


if __name__ == "__main__":
    model2validation = [
        ("sczbp/scz_only", "first_diagnosis"),
        ("sczbp/scz_only", "scz_diagnosis"),
        ("sczbp/scz_only", "bp_diagnosis"),
        ("sczbp/structured_text_xgboost_ddpm", "first_diagnosis"),
        ("sczbp/structured_text_xgboost_ddpm", "scz_diagnosis"),
        ("sczbp/structured_text_xgboost_ddpm", "bp_diagnosis"),
        ("sczbp/bp_only", "first_diagnosis"),
        ("sczbp/bp_only", "scz_diagnosis"),
        ("sczbp/bp_only", "bp_diagnosis"),
    ]

    m = scz_bp_auroc_by_outcome(model2validation_mapping=model2validation)  # type: ignore

    print(m)

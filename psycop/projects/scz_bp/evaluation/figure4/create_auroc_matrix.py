from pathlib import Path

import polars as pl
from confection import Config

from psycop.common.model_evaluation.binary.performance_by_type.auroc_by_outcome import (
    EvaluationFrame,
    auroc_by_outcome,
    plot_auroc_by_outcome,
)
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.scz_bp.dataset_description.scz_bp_table_one import SczBpTableOne


def scz_bp_validation_outcomes() -> list[EvaluationFrame]:
    cfg = Config().from_disk(Path(__file__).parent / "eval_config.cfg")
    meta_df = SczBpTableOne(cfg).get_filtered_prediction_times()

    meta_df = meta_df.rename(
        {
            "meta_scz_diagnosis_within_0_to_1825_days_max_fallback_0": "scz_diagnosis",
            "meta_bp_diagnosis_within_0_to_1825_days_max_fallback_0": "bp_diagnosis",
            "outc_first_scz_or_bp_within_0_to_1825_days_max_fallback_0": "first_diagnosis",
        }
    )

    meta_df = meta_df.with_columns(
        pl.concat_str(
            [pl.col("dw_ek_borger"), pl.col("timestamp").dt.strftime("%Y-%m-%d-%H-%M-%S")],
            separator="-",
        ).alias("pred_time_uuid")
    )

    return [
        EvaluationFrame(
            df=meta_df.select(["pred_time_uuid", "scz_diagnosis"]), outcome_col_name="scz_diagnosis"
        ),
        EvaluationFrame(
            df=meta_df.select(["pred_time_uuid", "first_diagnosis"]),
            outcome_col_name="first_diagnosis",
        ),
        EvaluationFrame(
            df=meta_df.select(["pred_time_uuid", "bp_diagnosis"]), outcome_col_name="bp_diagnosis"
        ),
    ]


if __name__ == "__main__":
    populate_baseline_registry()
    m = auroc_by_outcome(
        model_names=["sczbp/scz_only", "sczbp/structured_text_xgboost_ddpm", "sczbp/bp_only"],
        validation_outcomes=scz_bp_validation_outcomes(),
    )

    m = m.replace(
        {
            "sczbp/scz_only": "Schizophrenia",
            "sczbp/structured_text_xgboost_ddpm": "Any diagnosis",
            "sczbp/bp_only": "Bipolar disorder",
            "scz_diagnosis": "Schizophrenia",
            "first_diagnosis": "Any diagnosis",
            "bp_diagnosis": "Bipolar disorder",
        }
    )

    p = plot_auroc_by_outcome(m)
    print(m)

from pathlib import Path

import polars as pl
from confection import Config

from psycop.common.model_evaluation.binary.performance_by_type.auroc_by_outcome import (
    EvaluationFrame,
    auroc_by_outcome,
)
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.scz_bp.dataset_description.scz_bp_table_one import SczBpTableOne
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR


def scz_bp_validation_outcomes() -> list[EvaluationFrame]:
    cfg = Config().from_disk(Path(__file__).parent / "eval_config.cfg")
    meta_df = SczBpTableOne(cfg).get_filtered_prediction_times()

    meta_df = meta_df.rename(
        {
            "meta_scz_diagnosis_within_0_to_1825_days_max_fallback_0": "Schizophrenia",
            "meta_bp_diagnosis_within_0_to_1825_days_max_fallback_0": "Bipolar disorder",
            "outc_first_scz_or_bp_within_0_to_1825_days_max_fallback_0": "First schizophrenia or bipolar disorder diagnosis",
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
            df=meta_df.select(["pred_time_uuid", "Schizophrenia"]), outcome_col_name="Schizophrenia"
        ),
        EvaluationFrame(
            df=meta_df.select(
                ["pred_time_uuid", "First schizophrenia or bipolar disorder diagnosis"]
            ),
            outcome_col_name="First schizophrenia or bipolar disorder diagnosis",
        ),
        EvaluationFrame(
            df=meta_df.select(["pred_time_uuid", "Bipolar disorder"]),
            outcome_col_name="Bipolar disorder",
        ),
    ]


if __name__ == "__main__":
    populate_baseline_registry()

    best_train_model_name_mapping = {
        "sczbp/scz_structured_text_xgboost": "Schizophrenia only ",
        "sczbp/structured_text_xgboost_ddpm": "Joint model",
        "sczbp/bp_structured_text_xgboost": "Bipolar disorder only",
    }

    best_test_model_name_mapping = {
        "sczbp/test_scz_structured_text_ddpm": "Schizophrenia only",
        "sczbp/test_tfidf_1000": "Joint model",
        "sczbp/test_bp_structured_text_ddpm": "Bipolar disorder only",
    }
    validation_outcomes = scz_bp_validation_outcomes()

    for split, mapping in {
        "Train": best_train_model_name_mapping,
        "Test": best_test_model_name_mapping,
    }.items():
        m = auroc_by_outcome(
            model_names=(list(mapping.keys())), validation_outcomes=validation_outcomes
        )
        m = m.replace(mapping)
        m["ci_low"] = m["ci_low"].round(2)
        m["ci_high"] = m["ci_high"].round(2)
        m["estimate"] = (
            m["estimate"].astype(str)
            + " ("
            + m["ci_low"].astype(str)
            + ", "
            + m["ci_high"].astype(str)
            + ")"
        )
        m = m.pivot(columns="validation_outcome", index="model_name", values="estimate")
        m = m.sort_index(ascending=False)
        # reverse order of columns to get scz first
        m = m[
            [
                "Schizophrenia",
                "First schizophrenia or bipolar disorder diagnosis",
                "Bipolar disorder",
            ]
        ]

        with (SCZ_BP_EVAL_OUTPUT_DIR / f"{split}_auroc_matrix.html").open("w") as f:
            m.to_html(f)

"""Table with models as rows and time from first positive to outcome at different
positive predicted rates as columns, stratified by BP and SCZ"""

from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR
from psycop.projects.scz_bp.evaluation.figure2.first_positive_prediction_to_outcome_by_ppr import (
    scz_bp_time_from_first_positive_prediction_to_outcome_table,
)

if __name__ == "__main__":
    pprs = [0.01, 0.02, 0.04, 0.06, 0.08]

    for diagnosis in ["scz", "bp"]:
        modality2experiment_mapping = modality2experiment = {
            "Structured + text + synthetic": f"sczbp/{diagnosis}_ddpm",
            "Structured + text": f"sczbp/{diagnosis}_structured_text_xgboost",
            "Structured only ": f"sczbp/{diagnosis}_structured_only-xgboost",
            "Text only": f"sczbp/{diagnosis}_tfidf_1000-xgboost",
        }

        table = scz_bp_time_from_first_positive_prediction_to_outcome_table(
            modality2experiment=modality2experiment_mapping,
            pprs=pprs,
            group_by_outcome=True,
            model_type=diagnosis,  # type: ignore
        )
        table = table[table["outcome"] == diagnosis.upper()].drop(columns="outcome")

        with (SCZ_BP_EVAL_OUTPUT_DIR / f"figure_2_{diagnosis}.html").open("w") as f:
            f.write(table.to_html())

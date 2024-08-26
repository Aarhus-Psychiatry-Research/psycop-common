from pathlib import Path

import polars as pl
from confection import Config

from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.scz_bp.dataset_description.scz_bp_table_one import SczBpTableOne
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR
from psycop.projects.scz_bp.evaluation.figure2.confusion_matrix import scz_bp_confusion_matrix_plot
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_age import (
    scz_bp_auroc_by_age,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_calendar_time import (
    scz_bp_auroc_by_quarter,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_sex import (
    scz_bp_auroc_by_sex,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_time_from_first_visit import (
    scz_bp_auroc_by_time_from_first_contact,
)
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_best_run_in_experiment,
)


def prepare_metadata(metadata_cfg: Config) -> pl.DataFrame:
    return (
        SczBpTableOne(metadata_cfg)
        .get_filtered_prediction_times()
        .rename(
            {
                "meta_scz_diagnosis_within_0_to_1825_days_max_fallback_0": "scz_diagnosis",
                "meta_bp_diagnosis_within_0_to_1825_days_max_fallback_0": "bp_diagnosis",
                "outc_first_scz_or_bp_within_0_to_1825_days_max_fallback_0": "first_diagnosis",
            }
        )
        .with_columns(
            pl.concat_str(
                [pl.col("dw_ek_borger"), pl.col("timestamp").dt.strftime("%Y-%m-%d-%H-%M-%S")],
                separator="-",
            ).alias("pred_time_uuids")
        )
        .select("pred_time_uuids", "scz_diagnosis", "bp_diagnosis", "first_diagnosis")
    )


if __name__ == "__main__":
    best_experiment = "sczbp/test_tfidf_1000"
    best_pos_rate = 0.04

    best_eval_ds = scz_bp_get_eval_ds_from_best_run_in_experiment(
        experiment_name=best_experiment, model_type="joint"
    )

    panels = [
        scz_bp_auroc_by_sex(eval_ds=best_eval_ds.model_copy()),
        scz_bp_auroc_by_age(eval_ds=best_eval_ds.model_copy()),
        scz_bp_auroc_by_time_from_first_contact(eval_ds=best_eval_ds.model_copy()),
        scz_bp_auroc_by_quarter(eval_ds=best_eval_ds.model_copy()),
    ]

    metadata_cfg = Config().from_disk(
        Path(__file__).parent.parent.parent / "dataset_description" / "eval_config.cfg"
    )
    metadata_df = prepare_metadata(metadata_cfg=metadata_cfg)

    eval_df = (
        best_eval_ds.to_polars()
        .join(metadata_df, on="pred_time_uuids", how="left", validate="1:1")
        .to_pandas()
    )

    scz_df = pl.from_pandas(eval_df).filter(pl.col("scz_diagnosis").is_not_null()).to_pandas()
    bp_df = pl.from_pandas(eval_df).filter(pl.col("bp_diagnosis").is_not_null()).to_pandas()

    panels.append(
        scz_bp_confusion_matrix_plot(
            y_true=scz_df["scz_diagnosis"],
            y_hat=scz_df["y_hat_probs"],
            positive_rate=best_pos_rate,
            actual_outcome_text="SCZ within 5 years",
            predicted_text="SCZ or BP within 5 years",
            add_auroc=True,
        )
    )
    panels.append(
        scz_bp_confusion_matrix_plot(
            y_true=bp_df["bp_diagnosis"],
            y_hat=bp_df["y_hat_probs"],
            positive_rate=best_pos_rate,
            actual_outcome_text="BP within 5 years",
            predicted_text="SCZ or BP within 5 years",
            add_auroc=True,
        )
    )

    grid = create_patchwork_grid(plots=panels, single_plot_dimensions=(5, 5), n_in_row=2)
    grid.savefig(SCZ_BP_EVAL_OUTPUT_DIR / "scz_bp_fig_3.png")

    
    for diagnosis in ["bp", "scz"]:
        best_experiment = f"sczbp/test_{diagnosis}_structured_text"
        best_eval_ds = scz_bp_get_eval_ds_from_best_run_in_experiment(
        experiment_name=best_experiment, model_type=diagnosis
        )

        panels = [
        scz_bp_auroc_by_sex(eval_ds=best_eval_ds.model_copy()),
        scz_bp_auroc_by_age(eval_ds=best_eval_ds.model_copy()),
        scz_bp_auroc_by_time_from_first_contact(eval_ds=best_eval_ds.model_copy()),
        scz_bp_auroc_by_quarter(eval_ds=best_eval_ds.model_copy()),
        ]
        
        grid = create_patchwork_grid(plots=panels, single_plot_dimensions=(5, 5), n_in_row=2)
        grid.savefig(SCZ_BP_EVAL_OUTPUT_DIR / f"robustness_{diagnosis}_test_set.png")


        
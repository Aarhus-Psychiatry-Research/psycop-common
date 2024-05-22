import pandas as pd

from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR
from psycop.projects.scz_bp.evaluation.figure2.auroc_by_data_type import (
    plot_scz_bp_auroc_by_data_type,
)
from psycop.projects.scz_bp.evaluation.figure2.confusion_matrix import scz_bp_confusion_matrix_plot
from psycop.projects.scz_bp.evaluation.figure2.first_positive_prediction_to_outcome import (
    plot_scz_bp_first_pred_to_event_stratified,
)
from psycop.projects.scz_bp.evaluation.figure2.sensitivity_by_time_to_event import (
    scz_bp_plot_sensitivity_by_time_to_event,
)
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_best_run_in_experiment,
)
from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import populate_scz_bp_registry

if __name__ == "__main__":
    populate_scz_bp_registry()
    modality2experiment_mapping = modality2experiment = {
        "Structured + text + synthetic": "sczbp/test_structured_text_ddpm",
        "Structured + text": "sczbp/test_structured_text",
        "Structured only ": "sczbp/test_structured_only",
        "Text only": "sczbp/test_tfidf_1000",
    }
    best_experiment = "sczbp/test_tfidf_1000"

    best_pos_rate = 0.04

    best_eval_ds = scz_bp_get_eval_ds_from_best_run_in_experiment(experiment_name=best_experiment)

    panel_a = plot_scz_bp_auroc_by_data_type(modality2experiment_mapping)
    panel_b = scz_bp_confusion_matrix_plot(
        eval_ds=best_eval_ds.model_copy(), positive_rate=best_pos_rate
    )
    panel_c = scz_bp_plot_sensitivity_by_time_to_event(
        eval_ds=best_eval_ds.model_copy(), ppr=best_pos_rate
    )
    panel_d = plot_scz_bp_first_pred_to_event_stratified(
        eval_ds=best_eval_ds.model_copy(), ppr=best_pos_rate
    )

    panels = [panel_a, panel_b, panel_c, panel_d]

    with pd.option_context("mode.chained_assignment", None):
        grid = create_patchwork_grid(plots=panels, single_plot_dimensions=(5, 5), n_in_row=2, start_letter_index=2)
    grid.savefig(SCZ_BP_EVAL_OUTPUT_DIR / f"fig_2_{best_experiment.split('/')[1]}.png")

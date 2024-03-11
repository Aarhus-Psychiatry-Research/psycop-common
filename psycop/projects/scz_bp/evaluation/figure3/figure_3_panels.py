from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
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

if __name__ == "__main__":
    best_experiment = "sczbp/text_only"
    best_pos_rate = 0.04

    best_eval_ds = scz_bp_get_eval_ds_from_best_run_in_experiment(experiment_name=best_experiment)

    panels = [
        scz_bp_auroc_by_sex(eval_ds=best_eval_ds.model_copy()),
        scz_bp_auroc_by_age(eval_ds=best_eval_ds.model_copy()),
        scz_bp_auroc_by_time_from_first_contact(eval_ds=best_eval_ds.model_copy()),
        scz_bp_auroc_by_quarter(eval_ds=best_eval_ds.model_copy()),
    ]
    grid = create_patchwork_grid(plots=panels, single_plot_dimensions=(5, 5), n_in_row=2)
    grid.savefig("scz_bp_fig_3.png")

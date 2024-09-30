from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR
from psycop.projects.scz_bp.evaluation.cost_benefit.cost_benefit_distributions import (
    sample_float_from_truncated_log_normal,
    sample_int_from_truncated_normal,
)
from psycop.projects.scz_bp.evaluation.cost_benefit.cost_benefit_model import (
    CostBenefitDistributions,
    cost_benefit_model,
)
from psycop.projects.scz_bp.evaluation.cost_benefit.cost_benefit_view import (
    plot_cost_benefit_by_ppr,
    plot_sampling_distributions,
)
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_best_run_in_experiment,
)


def cost_benefit_facade():
    eval_df = scz_bp_get_eval_ds_from_best_run_in_experiment(
        experiment_name="sczbp/test_scz_structured_text_ddpm", model_type="scz"
    )

    cost_benefit_distributions = CostBenefitDistributions(
        cost_of_intervention=sample_int_from_truncated_normal(mean=2000, std=200, lower_bound=500),
        efficiency_of_intervention=sample_float_from_truncated_log_normal(
            mean=0.5, lower_bound=0.1, upper_bound=0.7
        ),
        savings_from_prevented_outcome=sample_int_from_truncated_normal(
            mean=35_000, std=5000, lower_bound=1000
        ),
    )
    distribution_plots = plot_sampling_distributions(
        cost_benefit_distributions=cost_benefit_distributions
    )
    grid = create_patchwork_grid(
            plots=distribution_plots,
            single_plot_dimensions=(5, 5),
            n_in_row=2,
            first_letter_index=0,
        )
    grid.savefig(
        SCZ_BP_EVAL_OUTPUT_DIR / f"cost_benefit_distributions_scz_35000.png", dpi=300
    )

    cost_benefit_df = cost_benefit_model(
        eval_df=eval_df,
        cost_benefit_distributions=cost_benefit_distributions,
        pprs=[0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01],
        per_true_positive=False,
        min_alert_days=None,
    )

    p = plot_cost_benefit_by_ppr(df=cost_benefit_df, per_true_positive=False)
    p.save(SCZ_BP_EVAL_OUTPUT_DIR / "cost_benefit_35000.png", dpi=300)



if __name__ == "__main__":
    cost_benefit_facade()
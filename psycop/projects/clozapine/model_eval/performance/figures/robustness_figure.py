from pathlib import Path

import polars as pl
from wasabi import Printer

msg = Printer(timestamp=True)


from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.combine_text_structured_clozapine_outcome import (
    get_first_clozapine_prescription,
)
from psycop.projects.clozapine.loaders.demographics import birthdays, sex_female
from psycop.projects.clozapine.model_eval.performance.robustness.clozapine_robustness_by_age import (
    auroc_by_age_model,
    plotnine_auroc_by_age,
)
from psycop.projects.clozapine.model_eval.performance.robustness.clozapine_robustness_by_month import (
    auroc_by_month_model,
    plotnine_auroc_by_month,
)
from psycop.projects.clozapine.model_eval.performance.robustness.clozapine_robustness_by_sex import (
    auroc_by_sex_model,
    plotnine_auroc_by_sex,
)
from psycop.projects.clozapine.model_eval.performance.robustness.clozapine_sensitivity_by_time_to_event import (
    sensitivity_by_time_to_event,
)
from psycop.projects.clozapine.model_eval.utils import read_eval_df_from_disk


def create_auroc_patchwork_four_panel(
    eval_df: pl.DataFrame,
    birthdays: pl.DataFrame,
    sex_df: pl.DataFrame,
    outcome_timestamps: pl.DataFrame,
    save_dir: Path,
    single_plot_dimensions: tuple[float, float] = (6, 6),
):
    """
    Create a 4-panel AUROC patchwork:
        AUROC by age     | AUROC by sex
        AUROC by month   | Sensitivity to outcome
        (last row empty or center last)
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    plots = []
    any_plot_failed = False

    # 1. AUROC by Age
    try:
        plots.append(
            plotnine_auroc_by_age(
                auroc_by_age_model(df=eval_df, birthdays=birthdays, bins=[18, *range(20, 70, 10)]),
                title="AUROC by Age",
            )
        )
        msg.good("AUROC by age done")
    except Exception as e:
        msg.fail(f"AUROC by age failed: {e}")
        any_plot_failed = True

    # 2. AUROC by Sex
    try:
        plots.append(
            plotnine_auroc_by_sex(
                auroc_by_sex_model(df=eval_df, sex_df=sex_df), title="AUROC by Sex"
            )
        )
        msg.good("AUROC by sex done")
    except Exception as e:
        msg.fail(f"AUROC by sex failed: {e}")
        any_plot_failed = True

    # 3. AUROC by Month
    try:
        plots.append(
            plotnine_auroc_by_month(auroc_by_month_model(df=eval_df), title="AUROC by Month")
        )
        msg.good("AUROC by month done")
    except Exception as e:
        msg.fail(f"AUROC by month failed: {e}")
        any_plot_failed = True

    # 4. Sensitivity to Outcome
    try:
        plots.append(
            sensitivity_by_time_to_event(
                eval_dataset=eval_df,
                outcome_timestamps=outcome_timestamps,
                positive_rates=[0.01, 0.03, 0.05, 0.075, 0.1, 0.2],
            )
        )
        msg.good("Sensitivity to outcome done")
    except Exception as e:
        msg.fail(f"Sensitivity to outcome failed: {e}")
        any_plot_failed = True

    if not any_plot_failed:
        grid = create_patchwork_grid(
            plots=plots, single_plot_dimensions=single_plot_dimensions, n_in_row=2
        )

        png_path = save_dir / "clozapine_auroc_patchwork_4panel_log_reg.png"
        pdf_path = save_dir / "clozapine_auroc_patchwork_4panel_log_reg.pdf"

        grid.savefig(png_path)
        msg.good(f"Saved patchwork PNG to {png_path}")

        grid.savefig(pdf_path)
        msg.good(f"Saved patchwork PDF to {pdf_path}")


if __name__ == "__main__":
    experiment_name = (
        "clozapine hparam, structured_text_365d_lookahead, "
        "log_reg, 1 year lookbehind filter, 2025_random_split"
    )
    eval_dir = (
        f"E:/shared_resources/clozapine/eval_runs/" f"{experiment_name}_best_run_evaluated_on_test"
    )
    eval_df = read_eval_df_from_disk(eval_dir)

    save_dir = Path("E:/shared_resources/clozapine/eval/figures")

    sex_df = pl.from_pandas(sex_female())

    birthday = pl.from_pandas(birthdays())

    outcome_timestamps = pl.from_pandas(get_first_clozapine_prescription())

    create_auroc_patchwork_four_panel(
        eval_df=eval_df,
        birthdays=birthday,
        sex_df=sex_df,
        outcome_timestamps=outcome_timestamps,
        save_dir=save_dir,
    )

from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl
from wasabi import Printer

msg = Printer(timestamp=True)


from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.combine_text_structured_clozapine_outcome import (
    get_first_clozapine_prescription,
)
from psycop.projects.clozapine.loaders.demographics import birthdays, sex_female
from psycop.projects.clozapine.model_eval.performance.figures.first_pos_pred_to_event import (
    first_pos_pred_to_model,
    plotnine_first_true_pos_pred_to_event,
)
from psycop.projects.clozapine.model_eval.performance.robustness.clozapine_robustness_by_age import (
    auroc_by_age_model,
    plotnine_auroc_by_age,
)
from psycop.projects.clozapine.model_eval.performance.robustness.clozapine_robustness_by_month import (
    auroc_by_month_model,
    plotnine_auroc_by_month,
)
from psycop.projects.clozapine.model_eval.performance.robustness.clozapine_robustness_by_region import (
    auroc_by_region_model,
    plotnine_auroc_by_region,
)
from psycop.projects.clozapine.model_eval.performance.robustness.clozapine_robustness_by_sex import (
    auroc_by_sex_model,
    plotnine_auroc_by_sex,
)
from psycop.projects.clozapine.model_eval.utils import read_eval_df_from_disk


def create_auroc_patchwork_five_panel(
    eval_df: pl.DataFrame,
    birthday: pd.DataFrame,
    sex_df: pl.DataFrame,
    outcome_timestamps: pl.DataFrame,
    save_dir: Path,
    single_plot_dimensions: tuple[float, float] = (6.0, 5.5),
):
    """
    Create a 5-panel AUROC patchwork:
        [0] AUROC by age     | [1] AUROC by sex
        [2] AUROC by month   | [3] AUROC by region
        [4] First true positive → event (spans both columns)

    Exports both PNG and PDF.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    plots = []
    any_plot_failed = False

    try:
        plots.append(
            plotnine_auroc_by_age(
                auroc_by_age_model(
                    df=eval_df, birthdays=pl.from_pandas(birthday), bins=[18, *range(20, 70, 10)]
                )
            )
        )
        msg.good("AUROC by age done")
    except Exception as e:
        msg.fail(f"AUROC by age failed: {e}")
        any_plot_failed = True

    try:
        plots.append(plotnine_auroc_by_sex(auroc_by_sex_model(df=eval_df, sex_df=sex_df)))
        msg.good("AUROC by sex done")
    except Exception as e:
        msg.fail(f"AUROC by sex failed: {e}")
        any_plot_failed = True

    try:
        plots.append(plotnine_auroc_by_month(auroc_by_month_model(df=eval_df)))
        msg.good("AUROC by month done")
    except Exception as e:
        msg.fail(f"AUROC by month failed: {e}")
        any_plot_failed = True

    try:
        plots.append(plotnine_auroc_by_region(auroc_by_region_model(df=eval_df)))
        msg.good("AUROC by region done")
    except Exception as e:
        msg.fail(f"AUROC by region failed: {e}")
        any_plot_failed = True

    try:
        plots.append(
            plotnine_first_true_pos_pred_to_event(
                first_pos_pred_to_model(df=eval_df, outcome_timestamps=outcome_timestamps),
                title="Time from first true positive to event",
            )
            + pn.theme(figure_size=(5, 3))
        )
        msg.good("First true positive → event done")
    except Exception as e:
        msg.fail(f"First true positive → event failed: {e}")
        any_plot_failed = True

    if not any_plot_failed:
        grid = create_patchwork_grid(
            plots=plots, single_plot_dimensions=single_plot_dimensions, n_in_row=2
        )

        png_path = save_dir / "clozapine_auroc_patchwork_robustness.png"
        pdf_path = save_dir / "clozapine_auroc_patchwork_robustness.pdf"

        grid.savefig(png_path)
        msg.good(f"Saved patchwork PNG to {png_path}")

        grid.savefig(pdf_path)
        msg.good(f"Saved patchwork PDF to {pdf_path}")


if __name__ == "__main__":
    experiment_name = (
        "clozapine hparam, structured_text_365d_lookahead, "
        "xgboost, 1 year lookbehind filter, 2025_random_split"
    )
    eval_dir = (
        f"E:/shared_resources/clozapine/eval_runs/" f"{experiment_name}_best_run_evaluated_on_test"
    )
    eval_df = read_eval_df_from_disk(eval_dir)

    save_dir = Path("E:/shared_resources/clozapine/eval/figures")

    sex_df = pl.from_pandas(sex_female())

    birthday = birthdays()

    outcome_timestamps = pl.from_pandas(get_first_clozapine_prescription())

    create_auroc_patchwork_five_panel(
        eval_df=eval_df,
        birthday=birthday,
        sex_df=sex_df,
        outcome_timestamps=outcome_timestamps,
        save_dir=save_dir,
    )

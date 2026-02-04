from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from wasabi import Printer

msg = Printer(timestamp=True)


from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.clozapine.model_eval.performance.auroc import auroc_model, auroc_plot
from psycop.projects.clozapine.model_eval.performance.plotnine_confusion_matrix import (
    confusion_matrix_model,
    plotnine_confusion_matrix,
)
from psycop.projects.clozapine.model_eval.utils import read_eval_df_from_disk


def create_auroc_confmat_patchwork_two_experiments(
    experiments: Mapping[str, pd.DataFrame],
    positive_rate: float,
    output_path: Path,
    single_plot_dimensions: tuple[float, float] = (6, 6),
):
    """
    Create a patchwork comparing two experiments (XGBoost first).
    Exports both PNG and PDF.
    """
    # Ensure directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # --- enforce ordering: XGBoost first ---
    ordered_experiments = [
        ("XGBoost", experiments["XGBoost"]),
        ("Logistic regression", experiments["Logistic regression"]),
    ]

    plots = []
    any_plot_failed = False

    # ---- AUROC row (column headers as titles) ----
    for exp_name, eval_df in ordered_experiments:
        try:
            auroc = auroc_model(eval_df)
            p_auroc = auroc_plot(auroc, title=exp_name)  # column headline
            plots.append(p_auroc)
            msg.good(f"AUROC ({exp_name}) done")
        except Exception as e:
            msg.fail(f"AUROC ({exp_name}) failed: {e}")
            any_plot_failed = True

    # ---- Confusion matrix row (no titles) ----
    for _, eval_df in ordered_experiments:
        try:
            cm = confusion_matrix_model(eval_df, positive_rate=positive_rate)
            p_cm = plotnine_confusion_matrix(cm, title="")  # empty title
            plots.append(p_cm)
            msg.good("Confusion matrix done")
        except Exception as e:
            msg.fail(f"Confusion matrix failed: {e}")
            any_plot_failed = True

    if not any_plot_failed:
        # create patchwork grid
        grid = create_patchwork_grid(
            plots=plots, single_plot_dimensions=single_plot_dimensions, n_in_row=2
        )

        # Save PNG
        png_path = output_path / "clozapine_patchwork.png"
        grid.savefig(png_path)
        msg.good(f"Saved patchwork PNG to {png_path}")

        # Save PDF
        pdf_path = output_path / "clozapine_patchwork.pdf"
        grid.savefig(pdf_path)
        msg.good(f"Saved patchwork PDF to {pdf_path}")


if __name__ == "__main__":
    experiment_names = {
        "Logistic regression": (
            "clozapine hparam, structured_text_365d_lookahead, log_reg, 1 year lookbehind filter, 2025_random_split_best_run_evaluated_on_test"
        ),
        "XGBoost": (
            "clozapine hparam, structured_text_365d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split_best_run_evaluated_on_test"
        ),
    }

    positive_rate = 0.05

    experiments = {}
    for label, exp_name in experiment_names.items():
        eval_dir = f"E:/shared_resources/clozapine/eval_runs/" f"{exp_name}"
        experiments[label] = read_eval_df_from_disk(eval_dir).to_pandas()

    output_path = Path(
        "E:/shared_resources/clozapine/eval/figures/" "clozapine_auroc_conf_matrix_figure2.png"
    )

    create_auroc_confmat_patchwork_two_experiments(
        experiments=experiments, positive_rate=positive_rate, output_path=output_path
    )

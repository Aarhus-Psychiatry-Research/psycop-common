import patchworklib as pw
import polars as pl

from psycop.common.feature_generation.loaders.raw import sql_load
from psycop.common.model_evaluation.patchwork.patchwork_grid import print_a4_ratio
from psycop.projects.restraint.evaluation.figures.auroc import auroc_model, auroc_plot
from psycop.projects.restraint.evaluation.figures.confusion_matrix import (
    confusion_matrix_model,
    plotnine_confusion_matrix,
)
from psycop.projects.restraint.evaluation.figures.sensitivity_by_tte import (
    plotnine_sensitivity_by_tte,
    sensitivity_by_tte_model,
)
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk


def plot_grid(
    df: pl.DataFrame,
    outcome_timestamps: pl.DataFrame,
    first_letter_index: int,
    best_pos_rate: float,
    single_plot_dimensions: tuple[float, float] = (5.0, 4.5),
    add_subpanels_letters: bool = True,
) -> pw.Bricks:
    plots = [
        auroc_plot(auroc_model(df.to_pandas())),
        plotnine_confusion_matrix(
            confusion_matrix_model(df=df.to_pandas(), positive_rate=best_pos_rate)
        ),
        plotnine_sensitivity_by_tte(
            sensitivity_by_tte_model(df=df, outcome_timestamps=outcome_timestamps)
        ),
    ]

    print_a4_ratio(plots, single_plot_dimensions, 3)

    bricks = []

    for plot in plots:
        # Iterate here to catch errors while only a single plot is in scope
        # Makes debugging much easier
        bricks.append(pw.load_ggplot(plot, figsize=single_plot_dimensions))

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    rows = []

    for i in range(len(bricks)):
        # Add the letter
        if add_subpanels_letters:
            bricks[i].set_index(alphabet[first_letter_index:][i].upper(), size="xx-large")

        # Add it to the row
        rows.append(bricks[i])

    # Combine the rows
    patchwork = pw.stack(rows, operator="|")

    return patchwork


if __name__ == "__main__":
    restraint_type = "all"
    data_path = f"E:/shared_resources/restraint/eval_runs/restraint_{restraint_type}_tuning_v2_best_run_n_days"

    best_pos_rate = 0.05
    df = read_eval_df_from_disk(data_path)

    outcome_timestamps = pl.DataFrame(
        sql_load(
            "SELECT pred_times.dw_ek_borger, pred_time, first_mechanical_restraint as timestamp FROM fct.psycop_coercion_outcome_timestamps as pred_times LEFT JOIN fct.psycop_coercion_outcome_timestamps_2 as outc_times ON (pred_times.dw_ek_borger = outc_times.dw_ek_borger AND pred_times.datotid_start = outc_times.datotid_start)"
        ).drop_duplicates()
    )

    figure = plot_grid(
        df=df,
        outcome_timestamps=outcome_timestamps,
        first_letter_index=1,
        best_pos_rate=best_pos_rate,
    )

    figure.savefig(data_path + f"/restraint_{restraint_type}_ppr{best_pos_rate}_plot_grid.png")

from pathlib import Path
from psycop.common.feature_generation.loaders.raw import sql_load
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.restraint.evaluation.auroc import auroc_model, auroc_plot
import polars as pl
import plotnine as pn
import patchworklib as pw

from psycop.projects.restraint.evaluation.confusion_matrix import confusion_matrix_model, plotnine_confusion_matrix
from psycop.projects.restraint.evaluation.first_pos_pred_to_event import first_pos_pred_to_model, plotnine_first_pos_pred_to_event
from psycop.projects.restraint.evaluation.sensitivity_by_tte import plotnine_sensitivity_by_tte, sensitivity_by_tte_model

def plot_grid(
    eval_df: pl.DataFrame,
    outcome_timestamps: pl.DataFrame,
    first_letter_index: int,
) -> pw.Bricks:

    plots = [
        auroc_plot(auroc_model(eval_df.to_pandas())),
        plotnine_confusion_matrix(
            confusion_matrix_model(df=eval_df.to_pandas(), positive_rate=best_pos_rate)),
        plotnine_sensitivity_by_tte(
            sensitivity_by_tte_model(df=eval_df, outcome_timestamps=outcome_timestamps)
        ),
        plotnine_first_pos_pred_to_event(
            first_pos_pred_to_model(df=eval_df, outcome_timestamps=outcome_timestamps)
        )
    ]

    ggplots: list[pn.ggplot] = []
    for plot in plots:
        ggplots.append(plot)

    figure = create_patchwork_grid(
        plots=ggplots,
        single_plot_dimensions=(5, 4.5),
        n_in_row=2,
        first_letter_index=first_letter_index,
        panel_letter_size="xx-large",
    )
    return figure


if __name__ == "__main__":
    save_dir = Path(__file__).parent / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_experiment = "restraint_text_hyper"
    best_pos_rate = 0.05
    eval_df = MlflowClientWrapper().get_best_run_from_experiment(
            experiment_name=best_experiment, metric="all_oof_BinaryAUROC"
        ).eval_frame().frame
    outcome_timestamps = pl.DataFrame(sql_load("SELECT pred_times.dw_ek_borger, pred_time, first_mechanical_restraint as timestamp FROM fct.psycop_coercion_outcome_timestamps as pred_times LEFT JOIN fct.psycop_coercion_outcome_timestamps_2 as outc_times ON (pred_times.dw_ek_borger = outc_times.dw_ek_borger AND pred_times.datotid_start = outc_times.datotid_start)").drop_duplicates())

    figure = plot_grid(
        eval_df=eval_df,
        outcome_timestamps=outcome_timestamps,
        first_letter_index=1,
    )

    figure.savefig(save_dir / "plot_grid.png")

from pathlib import Path
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.restraint.evaluation.figures.plot_grid_main import plot_grid
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk
import polars as pl

def run_all_plots(df: pl.DataFrame, outcome_timestamps: pl.DataFrame, best_pos_rate: float, save_dir: str, run_name: str):

    
    save_path = Path(f"{save_dir}/{run_name}/figures/")
    save_path.mkdir(parents=True, exist_ok=True)

    main_grid = plot_grid(
        df=df,
        outcome_timestamps=mechanical_outcome_timestamps,
        first_letter_index=0,
        best_pos_rate=best_pos_rate,
    )


    main_grid.savefig(save_path / "main_grid.png")





if __name__ == "__main__":
    restraint_type = "mechanical"
    best_pos_rate = 0.05
    
    save_dir = f"E:/shared_resources/restraint/eval_runs"

    df = read_eval_df_from_disk(
        f"E:/shared_resources/restraint/eval_runs/restraint_{restraint_type}_tuning_30_days_best_run_30_days_x2"
    )

    mechanical_outcome_timestamps = pl.DataFrame(
        sql_load(
            "SELECT pred_times.dw_ek_borger, pred_time, first_mechanical_restraint as timestamp FROM fct.psycop_coercion_outcome_timestamps as pred_times LEFT JOIN fct.psycop_coercion_outcome_timestamps_2 as outc_times ON (pred_times.dw_ek_borger = outc_times.dw_ek_borger AND pred_times.datotid_start = outc_times.datotid_start)"
        ).drop_duplicates()
    )

    all_outcome_timestamps = pl.DataFrame(
        sql_load(
            "SELECT pred_times.dw_ek_borger, pred_time, outc_times.datotid_start_sei as timestamp FROM fct.psycop_coercion_outcome_timestamps as pred_times LEFT JOIN fct.psycop_coercion_outcome_timestamps_2 as outc_times ON (pred_times.dw_ek_borger = outc_times.dw_ek_borger AND pred_times.datotid_start = outc_times.datotid_start)"
        ).drop_duplicates()
    )

    run_all_plots(df=df, outcome_timestamps=mechanical_outcome_timestamps, best_pos_rate=0.05, save_dir=save_dir, run_name="restraint_split_tuning_v2_best_run_evaluated_on_test")
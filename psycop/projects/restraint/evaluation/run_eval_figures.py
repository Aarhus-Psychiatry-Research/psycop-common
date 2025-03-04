from pathlib import Path
import polars as pl
from plotnine import ggplot

from psycop.common.feature_generation.loaders.raw import sql_load
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.projects.restraint.evaluation.figures.auroc_by_age import auroc_by_age_model, plotnine_auroc_by_age
from psycop.projects.restraint.evaluation.figures.auroc_by_month import auroc_by_month_model, plotnine_auroc_by_month
from psycop.projects.restraint.evaluation.figures.auroc_by_sex import auroc_by_sex_model, plotnine_auroc_by_sex
from psycop.projects.restraint.evaluation.figures.auroc_by_week import auroc_by_weekday_model, plotnine_auroc_by_weekday
from psycop.projects.restraint.evaluation.figures.plot_grid_main import plot_grid
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk

def run_auroc_figures(df: pl.DataFrame, save_dir: Path, experiment: str):
    plotnine_auroc_by_age(
        auroc_by_age_model(df=df, birthdays=pl.from_pandas(birthdays()), bins=[18, *range(20, 70, 10)])
    ).save(save_dir / f"{experiment}_auroc_by_age.png") # type: ignore

    plotnine_auroc_by_month(auroc_by_month_model(df=df)).save(save_dir / f"{experiment}_auroc_by_month.png") # type: ignore

    plotnine_auroc_by_sex(auroc_by_sex_model(df=df, sex_df=pl.from_pandas(sex_female()))).save(save_dir / f"{experiment}_auroc_by_sex.png") # type: ignore

    plotnine_auroc_by_weekday(auroc_by_weekday_model(df=df)).save(save_dir / f"{experiment}_auroc_by_weekday.png") # type: ignore

def run_all_figures(df: pl.DataFrame, experiment: str, best_pos_rate: float, eval_dir: str):
    save_dir = Path(eval_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_auroc_figures(df=df, save_dir=save_dir, experiment=experiment)

    outcome_timestamps = pl.DataFrame(
        sql_load(
            "SELECT pred_times.dw_ek_borger, pred_time, first_mechanical_restraint as timestamp FROM fct.psycop_coercion_outcome_timestamps as pred_times LEFT JOIN fct.psycop_coercion_outcome_timestamps_2 as outc_times ON (pred_times.dw_ek_borger = outc_times.dw_ek_borger AND pred_times.datotid_start = outc_times.datotid_start)"
        ).drop_duplicates()
    )

    plot_grid(df=df, outcome_timestamps=outcome_timestamps, first_letter_index=1, best_pos_rate=best_pos_rate).savefig(save_dir / f"{experiment}_plot_grid.png")


if __name__ == "__main__":

    eval_dir = "E:/shared_resources/restraint/eval_runs/restraint_split_tuning_best_run_evaluated_on_test"
    df =  read_eval_df_from_disk(eval_dir)
    experiment = "restraint_split"
    best_pos_rate = 0.05
  
    run_all_figures(df=df, experiment=experiment, best_pos_rate=best_pos_rate, eval_dir=eval_dir)


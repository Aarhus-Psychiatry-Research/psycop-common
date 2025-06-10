from pathlib import Path
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.restraint.evaluation.figures.auroc_by_age import auroc_by_age_model, plotnine_auroc_by_age
from psycop.projects.restraint.evaluation.figures.auroc_by_region import auroc_by_region_model, plotnine_auroc_by_region
from psycop.projects.restraint.evaluation.figures.auroc_by_sex import auroc_by_sex_model, plotnine_auroc_by_sex
from psycop.projects.restraint.evaluation.figures.auroc_by_week import auroc_by_weekday_model, plotnine_auroc_by_weekday
from psycop.projects.restraint.evaluation.figures.plot_grid_main import plot_grid
from psycop.projects.restraint.evaluation.figures.predictor_importance import restraint_generate_feature_importance_table
from psycop.projects.restraint.evaluation.figures.sensitivity_by_outcome import plotnine_sensitivity_by_first_outcome, plotnine_sensitivity_by_outcome, sensitivity_by_first_outcome_model, sensitivity_by_outcome_model
from psycop.projects.restraint.evaluation.tables.restraint_predictor_describer import generate_feature_description_df
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk
import polars as pl
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.projects.restraint.feature_generation.modules.loaders.load_restraint_outcome_timestamps import load_restraint_outcome_timestamps

def run_paper_outputs(df: pl.DataFrame, outcome_timestamps: pl.DataFrame, best_pos_rate: float, save_dir: str, run_name: str):
    
    save_path = Path(f"{save_dir}/{run_name}/paper_outputs/")
    save_path.mkdir(parents=True, exist_ok=True)

    plot_grid(
        df=df,
        outcome_timestamps=outcome_timestamps,
        first_letter_index=0,
        best_pos_rate=best_pos_rate,
    ).savefig(save_path / "main_grid.png")

    plotnine_auroc_by_age(auroc_by_age_model(df=df, birthdays=pl.from_pandas(birthdays()), bins=[18, 25, 35, 45, 55])).save(save_path / "auroc_by_age.png")

    plotnine_auroc_by_sex(auroc_by_sex_model(df=df, sex_df=pl.from_pandas(sex_female()))).save(save_path / "auroc_by_sex.png")

    plotnine_auroc_by_weekday(auroc_by_weekday_model(df=df)).save(save_path / "auroc_by_weekday.png")

    plotnine_auroc_by_region(auroc_by_region_model(df=df)).save(save_path / "restraint_auroc_by_region.png")

    outcome_df = pl.from_pandas(load_restraint_outcome_timestamps())
    
    plotnine_sensitivity_by_first_outcome(sensitivity_by_first_outcome_model(df=df, outcome_df=outcome_df)).save(
         save_path / "restraint_sensitivity_by_first_outcome.png"
    )

    plotnine_sensitivity_by_outcome(sensitivity_by_outcome_model(df=df, outcome_df=outcome_df)).save(
         save_path / "restraint_sensitivity_by_outcome.png"
    )

    run = MlflowClientWrapper().get_best_run_from_experiment(run_name, metric="all_oof_BinaryAUROC")

    restraint_generate_feature_importance_table(
        pipeline=run.sklearn_pipeline(), clf_model_name="classifier"
    ).to_html(save_path / "predictor_importance.html")

    training_data = pl.read_parquet(run.get_config()["trainer"]["training_data"]["paths"][0]).drop(["dw_ek_borger", "timestamp", "prediction_time_uuid"])

    feature_description = generate_feature_description_df(df=training_data)
    data = feature_description.to_pandas().to_html()
    (save_path / "predictor_description.html").write_text(data)





if __name__ == "__main__":
    run_name = "restraint_all_tuning_v2_best_run_evaluated_on_test_mechanical"
    best_pos_rate = 0.01
    
    save_dir = "E:/shared_resources/restraint/eval_runs"

    df = read_eval_df_from_disk(save_dir + "/" + run_name)

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

    outcome_df = pl.from_pandas(load_restraint_outcome_timestamps())

    run_paper_outputs(df=df, outcome_timestamps=mechanical_outcome_timestamps, best_pos_rate=best_pos_rate, save_dir=save_dir, run_name=run_name)
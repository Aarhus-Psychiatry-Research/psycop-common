from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry

CONFIG_PATH = (
    Path(__file__).parent / "configs" / "fixed_hyperparams_secondary_temporal_validation.cfg"
)

# Earliest date available for training (i.e. the date of the first prediction).
# Used as the fixed training start for the "cumulative_training" analysis
# (expanding window from the beginning of the data), and as the lower bound
# for the "expanding_retrospective" analysis.
DATASET_START_DATE = "2014-01-01"
DATASET_START_YEAR = int(DATASET_START_DATE[2:4])

# Most recent year with a usable full evaluation window. Data is only
# available up to June 2025, and outcomes have a 180-day lookahead, so an
# evaluation window running all the way to 2024-12-31 could push outcome
# data past the cutoff. LAST_EVAL_YEAR_END_DATE truncates the final year's
# evaluation window to keep the lookahead inside the available data.
LAST_EVAL_YEAR = 24
LAST_EVAL_YEAR_END_DATE = "2024-12-16"


def evaluation_interval_for_year(eval_year: int) -> tuple[str, str]:
    """Full-calendar-year evaluation interval for `eval_year`.

    e.g. eval_year=17 -> ("2017-01-01", "2017-12-31"). The most recent
    available year (LAST_EVAL_YEAR) ends at LAST_EVAL_YEAR_END_DATE instead
    of December 31st, so the 180-day outcome lookahead doesn't run past the
    data cutoff. Note: the "before" date_filter direction is exclusive, so
    the very last moment of the end date itself is excluded.
    """
    start_date = f"20{eval_year:02d}-01-01"
    if eval_year == LAST_EVAL_YEAR:
        return (start_date, LAST_EVAL_YEAR_END_DATE)
    return (start_date, f"20{eval_year:02d}-12-31")


def training_end_date_before_year(eval_year: int) -> str:
    """December 31st of the year immediately preceding `eval_year`.

    e.g. eval_year=17 -> "2016-12-31". Used as the training cutoff so
    training data never overlaps the evaluation year.
    """
    return f"20{eval_year - 1:02d}-12-31"


def eval_stratified_split(
    cfg: PsycopConfig,
    analysis_name: str,
    training_start_date: str,
    training_end_date: str,
    evaluation_interval: tuple[str, str],
) -> float:
    outcome_col_name: str = cfg.retrieve("trainer.outcome_col_name")
    preprocessing_pipeline = cfg.rem(
        "trainer.preprocessing_pipeline.*.temporal_col_filter"
    ).retrieve("trainer.preprocessing_pipeline")
    experiment_path: str = cfg.retrieve("project_info.experiment_path")

    # Setup for experiment
    cfg = (
        cfg.mut("logger.*.mlflow.experiment_name", f"inv_admission_temp_val, {analysis_name}")
        .mut(
            "logger.*.disk.run_path",
            f"{experiment_path}{analysis_name}/"
            f"{training_start_date}_{training_end_date}_{evaluation_interval[0]}_{evaluation_interval[1]}",
        )
        .add(
            "logger.*.mlflow.run_name",
            f"{analysis_name}_train_{training_start_date}_{training_end_date}"
            f"_eval_{evaluation_interval[0]}_{evaluation_interval[1]}",
        )
        .mut("trainer.@trainers", "split_trainer_separate_preprocessing")
        .rem("trainer.outcome_col_name")
        .rem("trainer.preprocessing_pipeline")
        .rem("trainer.n_splits")
    )

    # Python dicts are ordered, but we remove the timestamp column before generating predictions.
    # To filter based on timestamp, we need to add the filter before temporal columns are removed.
    # These shenanigans are needed to insert.
    # Handle training set setup
    cfg = (
        cfg.add("trainer.training_outcome_col_name", outcome_col_name)
        .add("trainer.training_preprocessing_pipeline", preprocessing_pipeline)
        .add(
            "trainer.training_preprocessing_pipeline.*.date_filter_start",
            {
                "@preprocessing": "date_filter",
                "column_name": "timestamp",
                "threshold_date": training_start_date,
                "direction": "after-inclusive",
            },
        )
        .add(
            "trainer.training_preprocessing_pipeline.*.date_filter_end",
            {
                "@preprocessing": "date_filter",
                "column_name": "timestamp",
                "threshold_date": training_end_date,
                "direction": "before",
            },
        )
        .add(
            "trainer.training_preprocessing_pipeline.*.temporal_col_filter",
            {"@preprocessing": "temporal_col_filter"},
        )
    )

    # Handle validation set setup
    cfg = (
        cfg.add(  # Handle validation dataset
            "trainer.validation_data", cfg.retrieve("trainer.training_data")
        )
        .add("trainer.validation_outcome_col_name", outcome_col_name)
        .add("trainer.validation_preprocessing_pipeline", preprocessing_pipeline)
        .add(
            "trainer.validation_preprocessing_pipeline.*.date_filter_start",
            {
                "@preprocessing": "date_filter",
                "column_name": "timestamp",
                "threshold_date": evaluation_interval[0],
                "direction": "after-inclusive",
            },
        )
        .add(
            "trainer.validation_preprocessing_pipeline.*.date_filter_end",
            {
                "@preprocessing": "date_filter",
                "column_name": "timestamp",
                "threshold_date": evaluation_interval[1],
                "direction": "before",
            },
        )
        .add(
            "trainer.validation_preprocessing_pipeline.*.temporal_col_filter",
            {"@preprocessing": "temporal_col_filter"},
        )
    )

    return train_baseline_model_from_cfg(cfg)


# ---------------------------------------------------------------------------
# Config generators, one per analysis described in the study design.
# Each returns a list of (training_start_date, training_end_date, evaluation_interval).
# ---------------------------------------------------------------------------


def cumulative_training_configs(eval_years: range) -> list[tuple[str, str, tuple[str, str]]]:
    """Secondary analysis 1: annually updated / cumulative models.

    For each evaluation year, train from the start of the dataset up to (not
    including) that year, using all data accumulated so far. The training
    window therefore expands forward in time as the evaluation year increases.
    Evaluation covers the full calendar year (see evaluation_interval_for_year).
    """
    return [
        (DATASET_START_DATE, training_end_date_before_year(y), evaluation_interval_for_year(y))
        for y in eval_years
    ]


def expanding_retrospective_window_configs(
    fixed_eval_year: int = LAST_EVAL_YEAR, window_lengths_years: range | None = None
) -> list[tuple[str, str, tuple[str, str]]]:
    """Secondary analysis 2: fixed evaluation year, increasingly long retrospective windows.

    Evaluation always happens on `fixed_eval_year` (default 2024), over the
    full calendar year (see evaluation_interval_for_year). The training
    window ends just before that year and extends backward one year at a
    time, so training-set length grows while the evaluation set stays fixed.
    `window_lengths_years` gives the number of years included in each
    training window; it defaults to 1 year up to as many years as are
    available back to DATASET_START_DATE (e.g. eval year 2024 with dataset
    start 2014 -> windows of 1 up to 10 years, the last one starting exactly
    at 2014-01-01).
    """
    max_window_years = fixed_eval_year - DATASET_START_YEAR
    if window_lengths_years is None:
        window_lengths_years = range(1, max_window_years + 1)

    training_end_date = training_end_date_before_year(fixed_eval_year)
    evaluation_interval = evaluation_interval_for_year(fixed_eval_year)
    return [
        (f"20{fixed_eval_year - n_years}-01-01", training_end_date, evaluation_interval)
        for n_years in window_lengths_years
    ]


def recency_training_configs(eval_years: range) -> list[tuple[str, str, tuple[str, str]]]:
    """Secondary analysis 3: recency-based models.

    Train exclusively on the single year immediately preceding the
    evaluation year (e.g. eval 2017 -> train on 2016 only). Evaluation
    covers the full calendar year (see evaluation_interval_for_year).
    """
    return [
        (f"20{y - 1}-01-01", training_end_date_before_year(y), evaluation_interval_for_year(y))
        for y in eval_years
    ]


ANALYSES = {
    "cumulative_training": lambda: cumulative_training_configs(range(21, 25)),
    "expanding_retrospective": lambda: expanding_retrospective_window_configs(),
    "recency": lambda: recency_training_configs(range(15, 25)),
}


if __name__ == "__main__":
    populate_baseline_registry()

    ANALYSES_TO_RUN = ["cumulative_training", "expanding_retrospective", "recency"]

    aurocs = {}
    for analysis_name in ANALYSES_TO_RUN:
        configs = ANALYSES[analysis_name]()
        for training_start_date, training_end_date, evaluation_interval in configs:
            aurocs[(analysis_name, training_start_date, training_end_date, evaluation_interval)] = (
                eval_stratified_split(
                    PsycopConfig().from_disk(CONFIG_PATH),
                    analysis_name=analysis_name,
                    training_start_date=training_start_date,
                    training_end_date=training_end_date,
                    evaluation_interval=evaluation_interval,
                )
            )

    print(aurocs)

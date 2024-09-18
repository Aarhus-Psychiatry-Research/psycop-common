import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import polars as pl

from psycop.common.cohort_definition import OutcomeTimestampFrame
from psycop.common.feature_generation.data_checks.flattened.feature_describer_tsflattener_v2 import (
    generate_feature_description_df,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits_to_psychiatry
from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    EvalFrame,
    MlflowClientWrapper,
    PsycopMlflowRun,
)
from psycop.common.model_evaluation.markdown.md_objects import (
    MarkdownArtifact,
    MarkdownFigure,
    MarkdownTable,
    create_supplementary_from_markdown_artifacts,
)
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.projects.t2d_bigdata.cohort_examination.filtering_flowchart import (
    filtering_flowchart_facade,
)
from psycop.projects.t2d_bigdata.cohort_examination.incidence_by_time.facade import (
    incidence_by_time_facade,
)
from psycop.projects.t2d_bigdata.cohort_examination.table_one.facade import table_one_facade
from psycop.projects.t2d_bigdata.feature_generation.cohort_definition.t2d_bigdata_cohort_definer import (
    t2d_bigdata_outcome_timestamps,
    t2d_bigdata_pred_filtering,
    t2d_bigdata_pred_times,
)
from psycop.projects.t2d_bigdata.model_evaluation.single_run.performance_by_ppr.model import (
    performance_by_ppr_model,
)
from psycop.projects.t2d_bigdata.model_evaluation.single_run.performance_by_ppr.view import (
    performance_by_ppr_view,
)
from psycop.projects.t2d_bigdata.model_evaluation.single_run.single_run_main import single_run_main
from psycop.projects.t2d_bigdata.model_evaluation.single_run.single_run_robustness import (
    single_run_robustness,
)


class T2DBigDataArtifactFacade(Protocol):
    def __call__(self, output_dir: Path) -> None:
        ...


def _markdown_artifacts_facade(
    output_path: Path,
    outcome_label: str,
    eval_df: EvalFrame,
    outcome_timestamps: OutcomeTimestampFrame,
    sex_df: pl.DataFrame,
    all_visits_df: pl.DataFrame,
    birthdays_df: pl.DataFrame,
    estimator_type: str,
    primary_pos_proportion: float,
    pos_proportions: Sequence[float],
    lookahead_years: int,
    first_letter_index: int,
) -> Sequence[MarkdownArtifact]:
    # Main figure
    main_figure_output_path = output_path / f"{outcome_label}_main_figure.png"
    main_figure = single_run_main(
        eval_frame=eval_df,
        desired_positive_rate=primary_pos_proportion,
        outcome_label=outcome_label,
        outcome_timestamps=outcome_timestamps,
        first_letter_index=first_letter_index,
    )
    main_figure.savefig(main_figure_output_path)

    # Robustness figure
    robustness_figure_output_path = output_path / f"{outcome_label}_robustness_figure.png"
    robustness_figure = single_run_robustness(
        eval_frame=eval_df, sex_df=sex_df, all_visits_df=all_visits_df, birthdays=birthdays_df
    )
    robustness_figure.savefig(robustness_figure_output_path)

    # Performance by PPR
    performance_by_ppr_output_path = output_path / f"{outcome_label}_performance_by_ppr.csv"
    performance_by_ppr_table = performance_by_ppr_view(
        performance_by_ppr_model(
            eval_df=eval_df, positive_rates=pos_proportions, outcome_timestamps=outcome_timestamps
        ),
        outcome_label=outcome_label,
    )
    performance_by_ppr_table.write_csv(performance_by_ppr_output_path)

    pos_rate_percent = f"{int(primary_pos_proportion * 100)}"
    artifacts = [
        MarkdownFigure(
            title=f"Performance of {estimator_type} at a {pos_rate_percent} predicted positive rate with {lookahead_years} years of lookahead",
            file_path=main_figure_output_path,
            description=f"**A**: Receiver operating characteristics (ROC) curve. **B**: Confusion matrix. PPV: Positive predictive value. NPV: Negative predictive value. **C**: Sensitivity by months from prediction time to event, stratified by desired predicted positive rate (PPR). Note that the numbers do not match those in Table 1, since all prediction times with insufficient lookahead distance have been dropped. **D**: Time (years) from the first positive prediction to the patient having developed {outcome_label} at a {pos_rate_percent}% predicted positive rate (PPR). The dashed line represents the median time, and the solid line represents the prediction time.",
            relative_to_path=output_path,
        ),
        MarkdownFigure(
            title=f"Robustness of {estimator_type} at a {pos_rate_percent} predicted positive rate with {lookahead_years} years of lookahead",
            file_path=robustness_figure_output_path,
            description="Robustness of the model across stratifications. Blue line is the area under the receiver operating characteristics curve. Grey bars represent the proportion of visits that are present in each group. Error bars are 95%-confidence intervals from 100-fold bootstrap.",
            relative_to_path=output_path,
        ),
        MarkdownTable.from_filepath(
            title=f"Performance of {estimator_type} with {lookahead_years} years of lookahead by predicted positive rate (PPR). Numbers are physical contacts.",
            table_path=performance_by_ppr_output_path,
            description=f"""**Predicted positive rate**: The proportion of contacts predicted positive by the model. Since the model outputs a predicted probability, this is a threshold set by us.
**True prevalence**: The proportion of contacts that qualified for type 2 diabetes within the lookahead window.
**PPV**: Positive predictive value.
**NPV**: Negative predictive value.
**FPR**: False positive rate.
**FNR**: False negative rate.
**TP**: True positives. Numbers are service contacts.
**TN**: True negatives. Numbers are service contacts.
**FP**: False positives. Numbers are service contacts.
**FN**: False negatives. Numbers are service contacts.
**% of all {outcome_label} captured**: Percentage of all patients who developed {outcome_label}, who had at least one positive prediction.
**Median years from first positive to {outcome_label}**: For all patients with at least one true positive, the number of days from their first positive prediction to being labelled as {outcome_label}.
            """,
        ),
    ]

    return artifacts


def single_run_facade(output_path: Path, run: PsycopMlflowRun) -> None:
    cfg = run.get_config()
    eval_frame = run.eval_frame()

    lookahead_days_str = re.findall(r".+_to_(\d+)_days.+", cfg["trainer"]["outcome_col_name"])[0]
    lookahead_years = int(int(lookahead_days_str) / 365)
    estimator_type = cfg["trainer"]["task"]["task_pipe"]["sklearn_pipe"]["*"]["model"][
        "@estimator_steps"
    ]

    artifacts = _markdown_artifacts_facade(
        outcome_label="T2D-bigdata",
        eval_df=eval_frame,
        outcome_timestamps=t2d_bigdata_outcome_timestamps(),
        sex_df=pl.from_pandas(sex_female()),
        all_visits_df=pl.from_pandas(physical_visits_to_psychiatry()),
        birthdays_df=pl.from_pandas(birthdays()),
        output_path=output_path,
        estimator_type=estimator_type,
        primary_pos_proportion=0.05,
        pos_proportions=[0.01, 0.05, 0.1, 0.2],
        lookahead_years=lookahead_years,
        first_letter_index=0,
    )

    markdown_text = create_supplementary_from_markdown_artifacts(
        artifacts=artifacts,
        first_table_index=1,
        table_title_prefix="Table",
        first_figure_index=1,
        figure_title_prefix="Figure",
    )

    (output_path / "Report.md").write_text(markdown_text)

    non_markdown_artifacts: Sequence[T2DBigDataArtifactFacade] = [
        lambda output_dir: table_one_facade(run=run, output_dir=output_dir),
        lambda output_dir: incidence_by_time_facade(output_dir=output_dir),
        lambda output_dir: filtering_flowchart_facade(
            prediction_time_bundle=t2d_bigdata_pred_filtering(), run=run, output_dir=output_dir
        ),
    ]
    for artifact in non_markdown_artifacts:
        artifact(output_path)

    resolved_cfg = BaselineSchema(**BaselineRegistry.resolve(cfg))
    training_data: pl.DataFrame = (
        resolved_cfg.trainer.training_data.load()  # type: ignore
        .collect()
        .drop(cfg["trainer"]["group_col_name"], "timestamp", cfg["trainer"]["uuid_col_name"])
    )
    feature_description = generate_feature_description_df(df=training_data)
    data = feature_description.to_pandas().to_html()
    (output_path / "feature_description.html").write_text(data)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    run_name = "T2D-bigdata layer 1, base"
    run = MlflowClientWrapper().get_best_run_from_experiment(
        experiment_name="T2D-bigdata, h, l-2, XGB", metric="all_oof_BinaryAUROC"
    )
    output_dir = Path() / "outputs" / run.name
    output_dir.mkdir(exist_ok=True, parents=True)
    single_run_facade(output_dir, run)

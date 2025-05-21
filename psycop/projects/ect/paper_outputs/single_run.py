import pickle
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from confection import Config
from sklearn.pipeline import Pipeline

import polars as pl

from psycop.common.feature_generation.data_checks.flattened.feature_describer_tsflattener_v2 import (
    generate_feature_description_df,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits_to_psychiatry
from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    EvalFrame,
)
from psycop.common.model_evaluation.markdown.md_objects import (
    MarkdownArtifact,
    MarkdownFigure,
    MarkdownTable,
    create_supplementary_from_markdown_artifacts,
)
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.types.validated_frame import ValidatedFrame
from psycop.projects.ect.cohort_examination.filtering_flowchart import filtering_flowchart_facade
from psycop.projects.ect.cohort_examination.incidence_by_time.facade import incidence_by_time_facade
from psycop.projects.ect.cohort_examination.table_one.facade import table_one_facade
from psycop.projects.ect.feature_generation.cohort_definition.ect_cohort_definition import (
    ect_pred_filtering,
)
from psycop.projects.ect.model_evaluation.auroc_by.roc_by_multiple_runs_model import (
    ExperimentWithNames,
)
from psycop.projects.ect.model_evaluation.feature_importance import (
    ect_feature_importance_table_facade,
)
from psycop.projects.ect.model_evaluation.performance_by_ppr.model import performance_by_ppr_model
from psycop.projects.ect.model_evaluation.performance_by_ppr.view import performance_by_ppr_view
from psycop.projects.ect.model_evaluation.single_run_main import single_run_main
from psycop.projects.ect.model_evaluation.single_run_robustness import single_run_robustness
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk


class ECTArtifactFacade(Protocol):
    def __call__(self, output_dir: Path) -> None: ...

def _markdown_artifacts_facade(
    output_path: Path,
    outcome_label: str,
    main_eval_df: pl.DataFrame,
    group_auroc_experiments: ExperimentWithNames,
    sex_df: pl.DataFrame,
    all_visits_df: pl.DataFrame,
    birthdays_df: pl.DataFrame,
    estimator_type: str,
    primary_pos_proportion: float,
    pos_proportions: Sequence[float],
    lookahead_days: int,
    first_letter_index: int,
) -> Sequence[MarkdownArtifact]:
    # Main figure
    main_figure_output_path = output_path / f"{outcome_label}_main_figure.png"
    
    main_figure = single_run_main(
        eval_df=main_eval_df,
        group_auroc_experiments=group_auroc_experiments,
        desired_positive_rate=primary_pos_proportion,
        outcome_label=outcome_label,
        first_letter_index=first_letter_index,
    )
    main_figure.savefig(main_figure_output_path)
    
    eval_frame = EvalFrame(frame=main_eval_df, allow_extra_columns=True)

    # Robustness figure
    robustness_figure_output_path = output_path / f"{outcome_label}_robustness_figure.png"
    robustness_figure = single_run_robustness(
        eval_frame=eval_frame, sex_df=sex_df, all_visits_df=all_visits_df, birthdays=birthdays_df
    )
    robustness_figure.savefig(robustness_figure_output_path)

    # Performance by PPR
    performance_by_ppr_output_path = output_path / f"{outcome_label}_performance_by_ppr.csv"
    performance_by_ppr_table = performance_by_ppr_view(
        performance_by_ppr_model(eval_df=eval_frame, positive_rates=pos_proportions),
        outcome_label=outcome_label,
    )
    performance_by_ppr_table.write_csv(performance_by_ppr_output_path)

    pos_rate_percent = f"{int(primary_pos_proportion * 100)}"
    artifacts = [
        MarkdownFigure(
            title=f"Performance of {estimator_type} at a {pos_rate_percent}% predicted positive rate with {lookahead_days} days of lookahead",
            file_path=main_figure_output_path,
            description=f"**A**: Receiver operating characteristics (ROC) curve for each feature set. The model using the feature set with the highest AUROC was used for figures B-D with a predicted positive rate of {pos_rate_percent}%. **B**: Confusion matrix. PPV: Positive predictive value. NPV: Negative predictive value. **C**: Sensitivity by months from prediction time to event, stratified by desired predicted positive rate (PPR).  **D**: Time (days) from the first positive prediction to the patient receiving treatment with {outcome_label} at a {pos_rate_percent}% predicted positive rate (PPR). The dashed line represents the median time, and the solid line represents the prediction time.",
            relative_to_path=output_path,
        ),
        MarkdownFigure(
            title=f"Robustness of {estimator_type} at a {pos_rate_percent} predicted positive rate with {lookahead_days} days of lookahead",
            file_path=robustness_figure_output_path,
            description="Robustness of the model across stratifications. Blue line is the area under the receiver operating characteristics curve. Grey bars represent the proportion of visits that are present in each group. Error bars are 95%-confidence intervals from 100-fold bootstrap.",
            relative_to_path=output_path,
        ),
        MarkdownTable.from_filepath(
            title=f"Performance of {estimator_type} with {lookahead_days} days of lookahead by predicted positive rate (PPR). Numbers are inpatient stays with a minimum duration of 7 days.",
            table_path=performance_by_ppr_output_path,
            description=f"""**Predicted positive rate**: The proportion of contacts predicted positive by the model. Since the model outputs a predicted probability, this is a threshold set by us.
**True prevalence**: The proportion of contacts that received treatment with ECT within the lookahead window.
**PPV**: Positive predictive value.
**NPV**: Negative predictive value.
**FPR**: False positive rate.
**FNR**: False negative rate.
**TP**: True positives. Numbers are service contacts.
**TN**: True negatives. Numbers are service contacts.
**FP**: False positives. Numbers are service contacts.
**FN**: False negatives. Numbers are service contacts.
**% of all {outcome_label} captured**: Percentage of all patients who received {outcome_label}, who had at least one positive prediction.
**Median days from first positive to {outcome_label}**: For all patients with at least one true positive prediction, the number of days from their first positive prediction to being labelled as {outcome_label}.
            """,
        ),
    ]

    return artifacts


def single_run_facade(
    output_path: Path, main_run_eval_df: pl.DataFrame, main_run_cfg: PsycopConfig, main_run_pipeline: Pipeline, group_auroc_experiments: ExperimentWithNames
) -> None:
    cfg = main_run_cfg
    eval_df = main_run_eval_df

    lookahead_days_str = re.findall(
        r".+_to_(\d+)_days.+", cfg["trainer"]["training_outcome_col_name"]
    )[0]
    lookahead_days = int(lookahead_days_str)
    estimator_type = cfg["trainer"]["task"]["task_pipe"]["sklearn_pipe"]["*"]["model"][
        "@estimator_steps"
    ]

    artifacts = _markdown_artifacts_facade(
        outcome_label="ECT",
        main_eval_df=eval_df,
        group_auroc_experiments=group_auroc_experiments,
        sex_df=pl.from_pandas(sex_female()),
        all_visits_df=pl.from_pandas(physical_visits_to_psychiatry()),
        birthdays_df=pl.from_pandas(birthdays()),
        output_path=output_path,
        estimator_type=estimator_type,
        primary_pos_proportion=0.02,
        pos_proportions=[0.01, 0.02, 0.03, 0.04],
        lookahead_days=lookahead_days,
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
    
    non_markdown_artifacts: Sequence[ECTArtifactFacade] = [
        lambda output_dir: table_one_facade(cfg=main_run_cfg, output_dir=output_dir),
        lambda output_dir: incidence_by_time_facade(output_dir=output_dir),
        # lambda output_dir: filtering_flowchart_facade(
        #     prediction_time_bundle=ect_pred_filtering(), cfg=main_run_cfg, output_dir=output_dir
        # ),
        lambda output_dir: ect_feature_importance_table_facade(pipeline=main_run_pipeline, output_dir=output_dir),
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

    structured_text_experiment = "ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter"
    structured_text_experiment_path = f"E:/shared_resources/ect/eval_runs/{structured_text_experiment}_best_run_evaluated_on_test"
    structured_text_df = read_eval_df_from_disk(structured_text_experiment_path)

    #read other dfs
    structured_only_experiment = "ECT-trunc-and-hp-structured_only-xgboost-no-lookbehind-filter"
    structured_only_experiment_path = f"E:/shared_resources/ect/eval_runs/{structured_only_experiment}_best_run_evaluated_on_test"
    structured_only_df = read_eval_df_from_disk(structured_only_experiment_path)

    text_only_experiment = "ECT-trunc-and-hp-text_only-xgboost-no-lookbehind-filter"
    text_only_experiment_path = f"E:/shared_resources/ect/eval_runs/{text_only_experiment}_best_run_evaluated_on_test"
    text_only_df = read_eval_df_from_disk(text_only_experiment_path)

    feature_set_eval_dfs = {
        "Structured only": structured_only_df,
        "Text only": text_only_df,
        "Structured + text": structured_text_df,
    }
    feature_sets = ["structured_only", "text_only", "structured_text"]

    for feature_set_name in feature_sets:

        experiment = f"ECT-trunc-and-hp-{feature_set_name}-xgboost-no-lookbehind-filter"
        experiment_path = f"E:/shared_resources/ect/eval_runs/{experiment}_best_run_evaluated_on_test"
        experiment_df = read_eval_df_from_disk(experiment_path)
        experiment_cfg = PsycopConfig(Config().from_disk(path=Path(experiment_path) / 'config.cfg'))
        with open(Path(experiment_path) / "sklearn_pipe.pkl", "rb") as f:
            experiment_pipeline = pickle.load(f)

        save_dir =  Path(experiment_path + "/figures") 
        save_dir.mkdir(parents=True, exist_ok=True)

        single_run_facade(
            output_path=save_dir,
            main_run_eval_df=experiment_df,
            main_run_cfg=experiment_cfg,
            main_run_pipeline=experiment_pipeline,
            group_auroc_experiments=ExperimentWithNames(feature_set_eval_dfs),
        )

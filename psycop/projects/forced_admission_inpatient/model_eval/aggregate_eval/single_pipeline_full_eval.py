from wasabi import Printer

from psycop.common.model_evaluation.markdown.md_objects import (
    MarkdownArtifact,
    MarkdownFigure,
    MarkdownTable,
)
from psycop.projects.forced_admission_inpatient.model_eval.model_description.performance.main_performance_figure import (
    fa_inpatient_create_main_performance_figure,
)
from psycop.projects.forced_admission_inpatient.model_eval.model_description.performance.performance_by_ppr import (
    fa_inpatient_output_performance_by_ppr,
)
from psycop.projects.forced_admission_inpatient.model_eval.model_description.robustness.main_robustness_figure import (
    fa_inpatient_create_main_robustness_figure,
)
from psycop.projects.forced_admission_inpatient.model_eval.model_permuation.train_test_different_lookaheads import (
    performance_by_lookahead_table,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)

msg = Printer(timestamp=True)


def fa_inpatient_create_markdown_artifacts(
    pipeline: ForcedAdmissionInpatientPipelineRun,
) -> list[MarkdownArtifact]:
    relative_to = pipeline.paper_outputs.artifact_path.parent
    pos_rate_percent = f"{int(pipeline.paper_outputs.pos_rate * 100)}%"
    lookahead_days = pipeline.inputs.cfg.preprocessing.pre_split.min_lookahead_days

    artifacts = [
        MarkdownFigure(
            title=f"Performance of {pipeline.model_type} at a {pos_rate_percent} predicted positive rate with {lookahead_days} days of lookahead",
            file_path=pipeline.paper_outputs.paths.figures
            / pipeline.paper_outputs.artifact_names.main_performance_figure,
            description="**A**: Receiver operating characteristics (ROC) curve. **B**: Confusion matrix. PPV: Positive predictive value. NPV: Negative predictive value. **C**: Sensitivity by months from prediction time to event, stratified by desired predicted positive rate (PPR). Note that the numbers do not match those in Table 1, since all prediction times with insufficient lookahead distance have been dropped.",
            relative_to_path=relative_to,
        ),
        MarkdownFigure(
            title=f"Robustness of {pipeline.model_type} at a {pos_rate_percent} predicted positive rate with {lookahead_days} years of lookahead",
            file_path=pipeline.paper_outputs.paths.figures
            / pipeline.paper_outputs.artifact_names.main_robustness_figure,
            description="Robustness of the model across stratifications. Blue line is the area under the receiver operating characteristics curve. Grey bars represent the proportion of visits that are present in each group. Error bars are 95%-confidence intervals from 100-fold bootstrap.",
            relative_to_path=relative_to,
        ),
        MarkdownTable(
            title=f"Performance of {pipeline.model_type} with {lookahead_days} days of lookahead by predicted positive rate (PPR). Numbers are physical contacts.",
            file_path=pipeline.paper_outputs.paths.tables
            / pipeline.paper_outputs.artifact_names.performance_by_ppr,
            description="""**Predicted positive rate**: The proportion of contacts predicted positive by the model. Since the model outputs a predicted probability, this is a threshold set by us.
**True prevalence**: The proportion of contacts that qualified for type 2 diabetes within the lookahead window.
**PPV**: Positive predictive value.
**NPV**: Negative predictive value.
**FPR**: False positive rate.
**FNR**: False negative rate.
**TP**: True positives. Numbers are service contacts.
**TN**: True negatives. Numbers are service contacts.
**FP**: False positives. Numbers are service contacts.
**FN**: False negatives. Numbers are service contacts.
**% of all FA captured**: Percentage of involuntary admissions, that were flagged by at least one true positive prediction.""",
        ),
    ]

    return artifacts  # type: ignore


def fa_inpatient_main_manuscript_eval(
    pipeline: ForcedAdmissionInpatientPipelineRun,
) -> list[MarkdownArtifact]:
    msg.divider(
        f"Evaluating {pipeline.inputs.cfg.preprocessing.pre_split.min_lookahead_days} - {pipeline.model_type} - {pipeline.name}",
    )
    msg.info(
        f"Generating main manuscript eval to {pipeline.paper_outputs.artifact_path}",
    )

    fa_inpatient_output_performance_by_ppr(run=pipeline)
    fa_inpatient_create_main_performance_figure(run=pipeline)
    fa_inpatient_create_main_robustness_figure(run=pipeline)
    performance_by_lookahead_table(
        run=pipeline,
        lookaheads_for_performance_eval=[30, 90, 180, 360],
    )

    return fa_inpatient_create_markdown_artifacts(pipeline=pipeline)


if __name__ == "__main__":
    from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    fa_inpatient_main_manuscript_eval(pipeline=get_best_eval_pipeline())  # type: ignore

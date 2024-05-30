from wasabi import Printer

from psycop.common.model_evaluation.markdown.md_objects import (
    MarkdownArtifact,
    MarkdownFigure,
    MarkdownTable,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.main_performance_figure import (
    t2d_create_main_performance_figure,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.performance_by_ppr import (
    t2d_output_performance_by_ppr,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.main_robustness_figure import (
    t2d_create_main_robustness_figure,
)
from psycop.projects.t2d.utils.pipeline_objects import T2DPipelineRun

msg = Printer(timestamp=True)


def t2d_create_markdown_artifacts(pipeline: T2DPipelineRun) -> list[MarkdownArtifact]:
    relative_to = pipeline.paper_outputs.artifact_path.parent
    pos_rate_percent = f"{int(pipeline.paper_outputs.pos_rate * 100)}%"
    lookahead_years = int(pipeline.inputs.cfg.preprocessing.pre_split.min_lookahead_days / 365)

    artifacts = [
        MarkdownFigure(
            title=f"Performance of {pipeline.model_type} at a {pos_rate_percent} predicted positive rate with {lookahead_years} years of lookahead",
            file_path=pipeline.paper_outputs.paths.figures
            / pipeline.paper_outputs.artifact_names.main_performance_figure,
            description="**A**: Receiver operating characteristics (ROC) curve. **B**: Confusion matrix. PPV: Positive predictive value. NPV: Negative predictive value. **C**: Sensitivity by months from prediction time to event, stratified by desired predicted positive rate (PPR). Note that the numbers do not match those in Table 1, since all prediction times with insufficient lookahead distance have been dropped. **D**: Time (years) from the first positive prediction to the patient having developed T2D at a 3% predicted positive rate (PPR).",
            relative_to_path=relative_to,
        ),
        MarkdownFigure(
            title=f"Robustness of {pipeline.model_type} at a {pos_rate_percent} predicted positive rate with {lookahead_years} years of lookahead",
            file_path=pipeline.paper_outputs.paths.figures
            / pipeline.paper_outputs.artifact_names.main_robustness_figure,
            description="Robustness of the model across stratifications. Blue line is the area under the receiver operating characteristics curve. Grey bars represent the proportion of visits that are present in each group. Error bars are 95%-confidence intervals from 100-fold bootstrap.",
            relative_to_path=relative_to,
        ),
        MarkdownTable.from_filepath(
            title=f"Performance of {pipeline.model_type} with {int(pipeline.inputs.cfg.preprocessing.pre_split.min_lookahead_days / 365)} years of lookahead by predicted positive rate (PPR). Numbers are physical contacts.",
            table_path=pipeline.paper_outputs.paths.tables
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
**% of all T2D captured**: Percentage of all patients who developed T2D, who had at least one positive prediction.
**Median years from first positive to T2D**: For all patients with at least one true positive, the number of days from their first positive prediction to being labelled as T2D.
            """,
        ),
    ]

    return artifacts  # type: ignore


def t2d_main_manuscript_eval(pipeline: T2DPipelineRun) -> list[MarkdownArtifact]:
    msg.divider(
        f"Evaluating {pipeline.inputs.cfg.preprocessing.pre_split.min_lookahead_days} - {pipeline.model_type} - {pipeline.name}"
    )
    msg.info(f"Generating main manuscript eval to {pipeline.paper_outputs.artifact_path}")

    t2d_output_performance_by_ppr(run=pipeline)
    t2d_create_main_performance_figure(run=pipeline)
    t2d_create_main_robustness_figure(run=pipeline)

    return t2d_create_markdown_artifacts(pipeline=pipeline)


if __name__ == "__main__":
    pass

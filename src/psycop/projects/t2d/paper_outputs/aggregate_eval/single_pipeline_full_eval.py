from psycop.projects.t2d.paper_outputs.aggregate_eval.md_objects import (
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
from psycop.projects.t2d.paper_outputs.run_pipeline_on_train import (
    get_test_pipeline_run,
)
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun
from wasabi import Printer

msg = Printer(timestamp=True)


def _t2d_create_markdown_artifacts(run: PipelineRun) -> list[MarkdownArtifact]:
    relative_to = run.paper_outputs.artifact_path.parent
    pos_rate_percent = f"{int(run.paper_outputs.pos_rate * 100)}%"
    lookahead_years = int(
        run.inputs.cfg.preprocessing.pre_split.min_lookahead_days / 365,
    )

    artifacts = [
        MarkdownFigure(
            title=f"Performance of {run.model_type} at a {pos_rate_percent} predicted positive rate with {lookahead_years} years of lookahead",
            file_path=run.paper_outputs.paths.figures
            / run.paper_outputs.artifact_names.main_performance_figure,
            description="**A**: Receiver operating characteristics (ROC) curve. **B**: Confusion matrix. PPV: Positive predictive value. NPV: Negative predictive value. **C**: Sensitivity by months from prediction time to event, stratified by desired predicted positive rate (PPR). Note that the numbers do not match those in Table 1, since all prediction times with insufficient lookahead distance have been dropped. **D**: Distribution of months from the first positive prediction to the patient fulfilling T2D-criteria at a 3% predicted positive rate (PPR).",
            relative_to=relative_to,
        ),
        MarkdownFigure(
            title=f"Robustness of {run.model_type} at a {pos_rate_percent} predicted positive rate with {lookahead_years} years of lookahead",
            file_path=run.paper_outputs.paths.figures
            / run.paper_outputs.artifact_names.main_robustness_figure,
            description="Robustness of the model across a variety of stratifications. Blue line is the area under the receiver operating characteristics curve. Grey bars represent the number of contacts in each group. Error bars are 95%-confidence intervals from 100-fold bootstrap.",
            relative_to=relative_to,
        ),
        MarkdownTable(
            title=f"Performance of {run.model_type} with {int(run.inputs.cfg.preprocessing.pre_split.min_lookahead_days / 365)} years of lookahead by predicted positive rate (PPR). Numbers are physical contacts.",
            file_path=run.paper_outputs.paths.tables
            / run.paper_outputs.artifact_names.performance_by_ppr,
            description="""**Predicted positive**: The proportion of contacts predicted positive by the model. Since the model outputs a predicted probability, this is a threshold set by us.
**True prevalence**: The proportion of contacts that qualified for type 2 diabetes within the lookahead window.
**PPV**: Positive predictive value.
**NPV**: Negative predictive value.
**FPR**: False positive rate.
**FNR**: False negative rate.
**TP**: True positives.
**TN**: True negatives.
**FP**: False positives.
**FN**: False negatives.
**Mean warning days**: For all patients with at least one true positive, the number of days from their first positive prediction to their diagnosis.
            """,
        ),
    ]

    return artifacts


def t2d_main_manuscript_eval(dev_pipeline: PipelineRun) -> list[MarkdownArtifact]:
    msg.divider(
        f"Evaluating {dev_pipeline.inputs.cfg.preprocessing.pre_split.min_lookahead_days} - {dev_pipeline.model_type} - {dev_pipeline.name}",
    )
    train_pipeline = get_test_pipeline_run(pipeline_to_train=dev_pipeline)

    msg.info(
        f"Generating main manuscript eval to {train_pipeline.paper_outputs.artifact_path}",
    )

    t2d_output_performance_by_ppr(run=train_pipeline)
    t2d_create_main_performance_figure(run=train_pipeline)
    t2d_create_main_robustness_figure(run=train_pipeline)

    return _t2d_create_markdown_artifacts(run=train_pipeline)


if __name__ == "__main__":
    pass

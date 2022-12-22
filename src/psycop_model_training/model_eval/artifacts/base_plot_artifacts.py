from pathlib import Path

from psycop_model_training.model_eval.artifacts.plots.performance_by_age import plot_performance_by_age
from psycop_model_training.model_eval.artifacts.plots.performance_over_time import plot_auc_by_time_from_first_visit, \
    plot_metric_by_calendar_time, plot_metric_by_cyclic_time, plot_metric_by_time_until_diagnosis
from psycop_model_training.model_eval.artifacts.plots.roc_auc import plot_auc_roc
from psycop_model_training.model_eval.artifacts.plots.sens_over_time import plot_sensitivity_by_time_to_outcome_heatmap
from psycop_model_training.model_eval.artifacts.tables.performance_by_threshold import \
    generate_performance_by_positive_rate_table
from psycop_model_training.model_eval.dataclasses import EvalDataset, ArtifactContainer
from psycop_model_training.utils.config_schemas.full_config import FullConfigSchema
from psycop_model_training.utils.utils import positive_rate_to_pred_probs
from sklearn.metrics import recall_score

def create_base_plot_artifacts(
    cfg: FullConfigSchema,
    eval_dataset: EvalDataset,
    save_dir: Path,
) -> list[ArtifactContainer]:
    """A collection of plots that are always generated."""
    pred_proba_percentiles = positive_rate_to_pred_probs(
        pred_probs=eval_dataset.y_hat_probs,
        positive_rate_thresholds=cfg.eval.positive_rate_thresholds,
    )

    lookahead_bins = cfg.eval.lookahead_bins

    return [
        ArtifactContainer(
            label="sensitivity_by_time_by_threshold",
            artifact=plot_sensitivity_by_time_to_outcome_heatmap(
                eval_dataset=eval_dataset,
                pred_proba_thresholds=pred_proba_percentiles,
                bins=lookahead_bins,
                save_path=save_dir / "sensitivity_by_time_by_threshold.png",
            ),
        ),
        ArtifactContainer(
            label="auc_by_time_from_first_visit",
            artifact=plot_auc_by_time_from_first_visit(
                eval_dataset=eval_dataset,
                bins=lookahead_bins,
                save_path=save_dir / "auc_by_time_from_first_visit.png",
            ),
        ),
        ArtifactContainer(
            label="auc_by_calendar_time",
            artifact=plot_metric_by_calendar_time(
                eval_dataset=eval_dataset,
                save_path=save_dir / "auc_by_calendar_time.png",
            ),
        ),
        ArtifactContainer(
            label="auc_by_hour_of_day",
            artifact=plot_metric_by_cyclic_time(
                eval_dataset=eval_dataset,
                bin_period="H",
                save_path=save_dir / "auc_by_hour_of_day.png",
            ),
        ),
        ArtifactContainer(
            label="auc_by_day_of_week",
            artifact=plot_metric_by_cyclic_time(
                eval_dataset=eval_dataset,
                bin_period="D",
                save_path=save_dir / "auc_by_day_of_week.png",
            ),
        ),
        ArtifactContainer(
            label="auc_by_month_of_year",
            artifact=plot_metric_by_cyclic_time(
                eval_dataset=eval_dataset,
                bin_period="M",
                save_path=save_dir / "auc_by_month_of_year.png",
            ),
        ),
        ArtifactContainer(
            label="auc_roc",
            artifact=plot_auc_roc(
                eval_dataset=eval_dataset,
                save_path=save_dir / "auc_roc.png",
            ),
        ),
        ArtifactContainer(
            label="recall_by_time_to_diagnosis",
            artifact=plot_metric_by_time_until_diagnosis(
                eval_dataset=eval_dataset,
                bins=lookahead_bins,
                metric_fn=recall_score,
                y_title="Sensitivity (recall)",
                save_path=save_dir / "recall_by_time_to_diagnosis.png",
            ),
        ),
        ArtifactContainer(
            label="performance_by_threshold",
            artifact=generate_performance_by_positive_rate_table(
                eval_dataset=eval_dataset,
                pred_proba_thresholds=pred_proba_percentiles,
                positive_rate_thresholds=cfg.eval.positive_rate_thresholds,
                output_format="df",
            ),
        ),
        ArtifactContainer(
            label="performance_by_age",
            artifact=plot_performance_by_age(
                eval_dataset=eval_dataset,
                save_path=save_dir / "performance_by_age.png",
            ),
        ),
    ]

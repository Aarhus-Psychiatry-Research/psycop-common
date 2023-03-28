from pathlib import Path

from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.model_eval.base_artifacts.plots.feature_importance import (
    plot_feature_importances,
)
from psycop_model_training.model_eval.base_artifacts.plots.performance_by_age import (
    plot_roc_auc_by_age,
)
from psycop_model_training.model_eval.base_artifacts.plots.performance_over_time import (
    plot_metric_by_calendar_time,
    plot_recall_by_calendar_time,
    plot_roc_auc_by_cyclic_time,
    plot_roc_auc_by_time_from_first_visit,
    plot_sensitivity_by_time_until_diagnosis,
)
from psycop_model_training.model_eval.base_artifacts.plots.precision_recall import (
    plot_precision_recall,
)
from psycop_model_training.model_eval.base_artifacts.plots.roc_auc import plot_auc_roc
from psycop_model_training.model_eval.base_artifacts.plots.time_from_first_positive_to_event import (
    plot_time_from_first_positive_to_event,
)
from psycop_model_training.model_eval.base_artifacts.tables.descriptive_stats_table import (
    DescriptiveStatsTable,
)
from psycop_model_training.model_eval.base_artifacts.tables.performance_by_threshold import (
    generate_performance_by_positive_rate_table,
)
from psycop_model_training.model_eval.base_artifacts.tables.tables import (
    generate_feature_importances_table,
    generate_selected_features_table,
)
from psycop_model_training.model_eval.dataclasses import (
    ArtifactContainer,
    EvalDataset,
    PipeMetadata,
)
from sklearn.metrics import recall_score


class BaseArtifactGenerator:
    """Generates the base artifacts, i.e. those that generalise across all
    projects, from an EvalDataset."""

    def __init__(
        self,
        cfg: FullConfigSchema,
        eval_ds: EvalDataset,
        pipe_metadata: PipeMetadata,
        save_dir: Path,
    ):
        self.cfg = cfg
        self.eval_ds = eval_ds
        self.save_dir = save_dir
        self.pipe_metadata = pipe_metadata

    def create_base_plot_artifacts(self) -> list[ArtifactContainer]:
        """A collection of plots that are always generated."""
        lookahead_bins = self.cfg.eval.lookahead_bins

        return [
            ArtifactContainer(
                label="auc_by_time_from_first_visit",
                artifact=plot_roc_auc_by_time_from_first_visit(
                    eval_dataset=self.eval_ds,
                    bins=lookahead_bins,
                    save_path=self.save_dir / "auc_by_time_from_first_visit.png",
                ),
            ),
            ArtifactContainer(
                label="auc_by_calendar_time",
                artifact=plot_metric_by_calendar_time(
                    eval_dataset=self.eval_ds,
                    save_path=self.save_dir / "auc_by_calendar_time.png",
                ),
            ),
            ArtifactContainer(
                label="auc_by_hour_of_day",
                artifact=plot_roc_auc_by_cyclic_time(
                    eval_dataset=self.eval_ds,
                    bin_period="H",
                    save_path=self.save_dir / "auc_by_hour_of_day.png",
                ),
            ),
            ArtifactContainer(
                label="auc_by_day_of_week",
                artifact=plot_roc_auc_by_cyclic_time(
                    eval_dataset=self.eval_ds,
                    bin_period="D",
                    save_path=self.save_dir / "auc_by_day_of_week.png",
                ),
            ),
            ArtifactContainer(
                label="auc_by_month_of_year",
                artifact=plot_roc_auc_by_cyclic_time(
                    eval_dataset=self.eval_ds,
                    bin_period="M",
                    save_path=self.save_dir / "auc_by_month_of_year.png",
                ),
            ),
            ArtifactContainer(
                label="auc_roc",
                artifact=plot_auc_roc(
                    eval_dataset=self.eval_ds,
                    save_path=self.save_dir / "auc_roc.png",
                ),
            ),
            ArtifactContainer(
                label="recall_by_time_to_diagnosis",
                artifact=plot_sensitivity_by_time_until_diagnosis(
                    eval_dataset=self.eval_ds,
                    bins=lookahead_bins,
                    roc_auc_score=recall_score,
                    y_title="Sensitivity (recall)",
                    save_path=self.save_dir / "recall_by_time_to_diagnosis.png",
                ),
            ),
            ArtifactContainer(
                label="performance_by_threshold",
                artifact=generate_performance_by_positive_rate_table(
                    eval_dataset=self.eval_ds,
                    output_format="df",
                ),
            ),
            ArtifactContainer(
                label="performance_by_age",
                artifact=plot_roc_auc_by_age(
                    eval_dataset=self.eval_ds,
                    save_path=self.save_dir / "performance_by_age.png",
                ),
            ),
            ArtifactContainer(
                label="time_from_first_positive_to_event",
                artifact=plot_time_from_first_positive_to_event(
                    eval_dataset=self.eval_ds,
                    bins=lookahead_bins,
                    save_path=self.save_dir / "time_from_first_positive_to_event.png",
                ),
            ),
            ArtifactContainer(
                label="precision_recall",
                artifact=plot_precision_recall(
                    eval_dataset=self.eval_ds,
                    save_path=self.save_dir / "precision_recall.png",
                ),
            ),
            ArtifactContainer(
                label="recall_by_calendar_time",
                artifact=plot_recall_by_calendar_time(
                    eval_dataset=self.eval_ds,
                    pos_rate=[0.95, 0.97, 0.99],
                    bins=lookahead_bins,
                    y_limits=(0, 0.5),
                    save_path=self.save_dir / "recall_by_calendar_time.png",
                ),
            ),
        ]

    def get_descriptive_stats_table_artifact(self) -> list[ArtifactContainer]:
        """Returns descriptive stats table artifact."""
        return [
            ArtifactContainer(
                label="descriptive_stats_table",
                artifact=DescriptiveStatsTable(
                    self.eval_ds,
                ).generate_descriptive_stats_table(),
            ),
        ]

    def get_feature_selection_artifacts(self) -> list[ArtifactContainer]:
        """Returns a list of artifacts related to feature selection."""
        return [
            ArtifactContainer(
                label="selected_features",
                artifact=generate_selected_features_table(
                    eval_dataset=self.pipe_metadata.selected_features,
                    output_format="df",
                ),
            ),
        ]

    def get_feature_importance_artifacts(self) -> list[ArtifactContainer]:
        """Returns a list of artifacts related to feature importance."""
        return [
            ArtifactContainer(
                label="feature_importances",
                artifact=plot_feature_importances(
                    feature_importance_dict=self.pipe_metadata.feature_importances,
                    save_path=self.save_dir / "feature_importances.png",
                ),
            ),
            ArtifactContainer(
                label="feature_importances",
                artifact=generate_feature_importances_table(
                    feature_importance_dict=self.pipe_metadata.feature_importances,
                    output_format="df",
                ),
            ),
        ]

    def get_all_artifacts(self) -> list[ArtifactContainer]:
        """Generates artifacts from an EvalDataset."""
        artifact_containers = self.create_base_plot_artifacts()

        if self.cfg.eval.descriptive_stats_table:
            artifact_containers += self.get_descriptive_stats_table_artifact()

        if self.pipe_metadata and self.pipe_metadata.feature_importances:
            artifact_containers += self.get_feature_importance_artifacts()

        if self.pipe_metadata and self.pipe_metadata.selected_features:
            artifact_containers += self.get_feature_selection_artifacts()

        return artifact_containers

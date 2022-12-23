from pathlib import Path

from artifacts.plots.performance_by_n_hba1c import plot_performance_by_n_hba1c

from psycop_model_training.model_eval.dataclasses import ArtifactContainer, EvalDataset


def create_custom_plot_artifacts(
    eval_dataset: EvalDataset,
    save_dir: Path,
) -> list[ArtifactContainer]:
    """A collection of plots that are only generated for your specific use
    case."""
    return [
        ArtifactContainer(
            label="performance_by_n_hba1c",
            artifact=plot_performance_by_n_hba1c(
                eval_dataset=eval_dataset,
                save_path=save_dir / "performance_by_n_hba1c.png",
            ),
        ),
    ]

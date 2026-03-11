from typing import Literal

from datetime import datetime

from psycop.common.cross_experiments.cross_project_catalogue import (
    ModelCatalogue,
    CROSS_EXPERIMENTS_BASE_PATH,
)
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)


def optimize_models_on_metric(
    projects: list[Literal["CVD", "ECT", "Restraint", "FAI", "SCZ_BP", "T2D"]],
    experiment_name: str,
    metric: str,
    max_fpr: float | None = None,
):

    catalogue = ModelCatalogue(projects=projects)
    tuning_cfgs = catalogue.get_hyperparameter_tuning_cfgs()

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{experiment_name}_{date_str}"
    experiment_path = CROSS_EXPERIMENTS_BASE_PATH + experiment_name

    for project, cfg in tuning_cfgs.items():
        print(f"Optimizing {project} on {metric}")

        n_trials = catalogue.project_getters[project].n_trials  # type: ignore
        n_jobs = catalogue.project_getters[project].n_jobs  # type: ignore
        project_name = f"{project}_{experiment_name}"
        project_path = f"{experiment_path}/{project_name}"

        cfg = cfg.mut("logger.*.mlflow.experiment_name", project_name).mut(
            "logger.*.disk_logger.run_path", project_path
        )
        cfg = cfg.mut("training.metric", metric)

        if max_fpr is not None:
            cfg = cfg.add("training.metric.max_fpr", max_fpr)

        OptunaHyperParameterOptimization().from_cfg(
            cfg=cfg,
            study_name=project_name,
            n_trials=n_trials,
            n_jobs=n_jobs,
            direction="maximize",
            catch=(),  # type: ignore
        )


if __name__ == "__main__":
    optimize_models_on_metric(
        projects=["Restraint"],
        experiment_name="all_projects_concentrated_auroc_tuning",
        metric="concentrated_binary_auroc",
    )

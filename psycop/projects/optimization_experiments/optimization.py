from datetime import datetime

from psycop.common.cross_experiments.cross_project_catalogue import (
    CROSS_EXPERIMENTS_BASE_PATH,
    ModelCatalogue,
)
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)
import mlflow

def optimize_models_on_metric(
    catalogue: ModelCatalogue, project: str, cfg: PsycopConfig, experiment_name: str, metric: str, max_fpr: float | None = None
):
    print(f"Optimizing {project} on {metric}")

    n_trials = catalogue.project_getters[project].n_trials  # type: ignore
    n_jobs = catalogue.project_getters[project].n_jobs  # type: ignore
    project_name = f"{project}_{experiment_name}"
    project_path = f"{CROSS_EXPERIMENTS_BASE_PATH}/{project_name}"


    cfg = cfg.mut("logger.*.mlflow.experiment_name", project_name).mut(
        "logger.*.disk_logger.run_path", project_path
    )

    cfg = cfg.mut("trainer.metric", {"@metrics": metric})

    if max_fpr is not None:
        cfg = cfg.add("trainer.metric.max_fpr", max_fpr)

    # mlflow.set_tracking_uri("http://localhost:5129")
    # mlflow.set_experiment(experiment_name=project_name)

    OptunaHyperParameterOptimization().from_cfg(
            cfg=cfg,
            study_name=project_name,
            n_trials=n_trials,
            n_jobs=n_jobs,
            direction="maximize",
            catch=(),  # type: ignore
            custom_populate_registry_fn=None,
    )

    print("hey")

if __name__ == "__main__":
    catalogue = ModelCatalogue(
        projects=["FAI"]
    )  # ["CVD", "ECT", "Restraint", "FAI", "SCZ_BP", "T2D"])
    tuning_cfgs = catalogue.get_hyperparameter_tuning_cfgs()

    metric = "binary_auroc"

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{metric}_tuning_{date_str}"
    # experiment_path = CROSS_EXPERIMENTS_BASE_PATH + experiment_name

    for project, cfg in tuning_cfgs.items():
        optimize_models_on_metric(catalogue, project, cfg, experiment_name, metric, max_fpr=None)


# Evaluate model on test set using a cfg logged to disk (to replicate model reported in the published paper 1/10/25)
from pathlib import Path

from confection import Config

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.cvd.model_training.populate_cvd_registry import populate_with_cvd_registry


def eval_geographic_test_set(
    cfg: PsycopConfig,
    experiment_name: str,
    test_run_name: str = "evaluated_on_test",
    ):

    test_run_experiment_name = f"{experiment_name}_best_run_{test_run_name}"
    test_run_path = "E:/shared_resources/" + "/cvd" + "/eval_runs/" + test_run_experiment_name

    if Path(test_run_path).exists():
        while True:
            response = input(
                f"This path '{test_run_path}' already exists. Do you want to potentially overwrite the contents of this folder with new feature sets? (yes/no): "
            )

            if response.lower() not in ["yes", "y", "no", "n"]:
                print("Invalid response. Please enter 'yes/y' or 'no/n'.")
            if response.lower() in ["no", "n"]:
                print("Process stopped.")
                return
            if response.lower() in ["yes", "y"]:
                print("Content of folder may be overwritten.")
                break

    train_baseline_model_from_cfg(cfg)


if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_cvd_registry()


    run_name = "CVD-replicate-model-in-paper"
    run_path = (
        f"E:/shared_resources/cvd/eval_runs/{run_name}_best_run_evaluated_on_test"
    )
    run_cfg = PsycopConfig(Config().from_disk(path=Path(run_path) / "config.cfg"))


    eval_geographic_test_set(run_cfg, run_name)

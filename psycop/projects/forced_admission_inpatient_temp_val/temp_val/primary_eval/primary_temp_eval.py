from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig


def eval_on_temporal_split(model: str, cfg: PsycopConfig):
    test_run_experiment_name = f"{model}_evaluated_on_full_temporal_split"

    test_run_path = (
        "E:/shared_resources/forced_admissions_inpatient_temp_val/"
        f"eval_runs/{test_run_experiment_name}"
    )

    if Path(test_run_path).exists():
        while True:
            response = input(f"Path '{test_run_path}' exists. Overwrite? (yes/no): ")

            if response.lower() in ["no", "n"]:
                print("Process stopped.")
                return

            if response.lower() in ["yes", "y"]:
                print("Content may be overwritten.")
                break

            print("Invalid response. Please enter yes/no.")

    cfg = (
        cfg.mut("logger.*.mlflow.experiment_name", test_run_experiment_name)
        .mut("logger.*.disk_logger.run_path", test_run_path)
        .mut("trainer.@trainers", "split_trainer_separate_preprocessing")
    )

    train_baseline_model_from_cfg(cfg)


if __name__ == "__main__":
    model = "primary_eval_xgboost_structured_features"
    cfg = PsycopConfig().from_disk(Path(__file__).parent / "configs" / f"{model}.cfg")

    eval_on_temporal_split(model=model, cfg=cfg)

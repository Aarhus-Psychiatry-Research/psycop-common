from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry

if __name__ == "__main__":
    populate_baseline_registry()

    experiments = [
        "ia_outpatient_all_features_training",
        "ia_outpatient_structured_features_training",
        "ia_outpatient_text_features_training",
    ]

    for experiment in experiments:
        cfg = (
            MlflowClientWrapper()
            .get_best_run_from_experiment(experiment_name=experiment, metric="all_oof_BinaryAUROC")
            .get_config()
        )

        train_baseline_model_from_cfg(cfg)

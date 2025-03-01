from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig


def train_with_score2(cfg: PsycopConfig):
    # Set run name
    cfg.mut("logger.*.mlflow.run_name", "T2D-bigdata, logistic regression, SCORE2")

    # Switch to logistic regression
    cfg.mut(
        "trainer.task.task_pipe.sklearn_pipe.*",
        {
            "imputer": {"@estimator_steps": "simple_imputation", "strategy": "mean"},
            "scaler": {"@estimator_steps": "standard_scaler"},
            "model": {"@estimator_steps": "logistic_regression", "max_iter": 1000},
        },
    )

    # Filter features by SCORE2 features
    features_to_keep = [
        "sex",
        "age",
        "ldl",
        "systolic",
        "smoking_categorical",
        "hdl",
        "total_cholesterol",
    ]
    cfg.mut(
        "trainer.preprocessing_pipeline.*",
        {"layer_selector": {"keep_matching": f".*({'|'.join(features_to_keep)}).*"}},
    )
    train_baseline_model_from_cfg(cfg)

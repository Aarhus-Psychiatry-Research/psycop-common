from pyexpat import features

import confection


def train_with_score2(cfg: confection.Config):
    # Set run name
    cfg["logger"]["*"]["mlflow"]["run_name"] = "CVD, logistic regression, SCORE2"

    # Switch to logistic regression
    cfg["trainer"]["task"]["task_pipe"]["sklearn_pipe"]["*"]["model"]["@estimator_steps"] = (
        "logistic_regression"
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
    cfg["trainer"]["preprocessing_pipeline"]["*"]["layer_selector"]["keep_matching"] = (
        f".*({'|'.join(features_to_keep)}).*"
    )

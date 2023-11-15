import optuna


def get_xgboost_hparams(trial: optuna.Trial) -> dict[str, float | str]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", low=100, high=1200, step=100),
        "alpha": trial.suggest_float("alpha", low=1e-8, high=1, step=None, log=True),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", low=1e-8, high=1, step=None, log=True
        ),
        "max_depth": trial.suggest_int("max_depth", low=1, high=10, step=1),
        "learning_rate": trial.suggest_float(
            "learning_rate", low=1e-8, high=0.3, step=None, log=True
        ),
        "gamma": trial.suggest_float("gamma", low=1e-8, high=1, step=None, log=True),
        "grow_policy": str(
            trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        ),
    }


def get_logistic_regression_hparams(trial: optuna.Trial) -> dict[str, float | str]:
    return {
        "C": trial.suggest_float("C", low=1e-5, high=1.0, step=None, log=True),
        "l1_ratio": trial.suggest_float(
            "l1_ratio", low=1e-8, high=1, step=None, log=True
        ),
    }


def get_simple_imputer_hparams(trial: optuna.Trial) -> dict[str, float | str]:
    return {
        "strategy": trial.suggest_categorical(
            "strategy", ["mean", "median", "most_frequent"]
        )
    }

from psycop.common.model_training_v2.config.config_utils import PsycopConfig


def elastic_net(cfg: PsycopConfig) -> PsycopConfig:
    # remove sklearn pipeline from config
    # add elastic net specific estimator steps: imputation, feature selection, model

    return (
        cfg.rem("trainer.task.task_pipe.sklearn_pipe.*.model")
        .add(
            "trainer.task.task_pipe.sklearn_pipe.*.scaler", {"@estimator_steps": "standard_scaler"}
        )
        .add(
            "trainer.task.task_pipe.sklearn_pipe.*.imputation",
            {"@estimator_steps": "simple_imputation", "strategy": "median"},
        )
        .add(
            "trainer.task.task_pipe.sklearn_pipe.*.feature_selection",
            {
                "@estimator_steps": "select_percentile",
                "score_function_name": "f_classif",
                "percentile": 32,
            },
        )
        .add(
            "trainer.task.task_pipe.sklearn_pipe.*.model",
            {
                "@estimator_steps": "logistic_regression",
                "penalty": "elasticnet",
                "solver": "saga",
                "C": 0.018535934517088258,
                "fit_intercept": True,
                "tol": 0.0001,
                "l1_ratio": 0.9263416322821825,
                "max_iter": 100,
                "intercept_scaling": True,
            },
        )
    )

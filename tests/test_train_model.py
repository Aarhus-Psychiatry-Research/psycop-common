from hydra import compose, initialize

from psycopt2d.train_model import main


def test_main_xgboost():
    """test main using xgboost."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(config_name="test_config.yaml", overrides=["+model=xgboost"])
        main(cfg)


def test_main_logistic_regression():
    """test main using logistic regresion."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name="test_config.yaml",
            overrides=["+model=logistic-regression"],
        )
        main(cfg)

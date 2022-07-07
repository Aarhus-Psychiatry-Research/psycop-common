from hydra import compose, initialize

from psycopt2d.train_model import main


def test_main():
    """test main."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(config_name="quick.yaml")
        main(cfg)

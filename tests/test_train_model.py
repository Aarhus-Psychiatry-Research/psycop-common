from hydra import compose, initialize

from psycopt2d.train_model import main


def test_main():
    """test main."""
    with initialize(version_base=None, config_path="./configs"):
        cfg = compose(config_name="test_basic_pipeline.yaml")
        main(cfg)

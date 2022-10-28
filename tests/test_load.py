"""Testing of loader functions."""
from hydra import compose, initialize

from psycopt2d.load import load_train_and_val_from_cfg
from psycopt2d.utils.configs import omegaconf_to_pydantic_objects


def test_load_lookbehind_exceeds_lookbehind_threshold():
    """Test that columns are dropped if their lookbehind are larger than the
    lookbehind threshold."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name="integration_config.yaml",
            overrides=["data.min_lookbehind_days=60"],
        )

        cfg = omegaconf_to_pydantic_objects(cfg)
        split_dataset = load_train_and_val_from_cfg(cfg)

        assert split_dataset.train.shape[1] == 7


def test_load_lookbehind_not_in_lookbehind_combination():
    """Test that columns are dropped if their lookbehind is not in the
    specified lookbehind combination list."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name="integration_config.yaml",
            overrides=["data.lookbehind_combination=[30]"],
        )

        cfg = omegaconf_to_pydantic_objects(cfg)

        split_dataset = load_train_and_val_from_cfg(cfg)

        assert split_dataset.train.shape[1] == 6

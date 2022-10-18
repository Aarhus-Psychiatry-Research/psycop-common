"""Testing of loader functions."""
from hydra import compose, initialize

from psycopt2d.load import load_train_and_val_from_cfg


def test_load_feat_lookbehind_exceeds_lookbehind_threshold():
    """Test that columns are dropped if their lookbehind are larger than the lookbehind threshold."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name="integration_testing.yaml",
            overrides=[
                "++data.min_lookbehind_days=90",
            ],
        )

        split_dataset = load_train_and_val_from_cfg(cfg)

        assert split_dataset.train.shape == (644, 7)


def test_load_feat_lookbehind_not_in_lookbehind_combination():
    """Test that columns are dropped if their lookbehind is not in the specified lookbehind combination list"""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name="integration_testing.yaml",
            overrides=[
                "++data.lookbehind_combination=[30]",
            ],
        )

        split_dataset = load_train_and_val_from_cfg(cfg)

        assert split_dataset.train.shape == (700, 6)

"""Testing of loader functions."""
from hydra import compose, initialize

from psycopt2d.load import load_train_and_val_from_file


def test_load_feat_lookbehind_larger_than_min_lookbehind():
    """Test that columns are dropped if their lookbehind are larger than
    min_lookbehind."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name="integration_testing.yaml",
            overrides=[
                "++data.min_lookbehind_days=90",
            ],
        )

        train, _ = load_train_and_val_from_file(cfg)

        assert train.shape == (669, 7)

"""Example of how to inspect a dataset using the configs."""
from psycop.common.model_training.config_schemas.conf_utils import (
    load_test_cfg_as_pydantic,
)
from psycop.common.model_training.data_loader.utils import (
    load_and_filter_train_from_cfg,
    load_train_raw,
)


def main():
    """Main."""
    config_file_name = "default_config.yaml"

    cfg = load_test_cfg_as_pydantic(config_file_name=config_file_name)
    load_train_raw(cfg=cfg)  # pylint: disable=unused-variable

    load_and_filter_train_from_cfg(  # pylint: disable=unused-variable
        cfg=cfg,
    )


if __name__ == "__main__":
    main()

"""Example of how to inspect a dataset using the configs."""
from psycopt2d.load import load_train_from_cfg
from psycopt2d.utils.config_schemas import load_cfg_as_pydantic


def main():
    """Main."""
    config_file_name = "default_config.yaml"

    cfg = load_cfg_as_pydantic(config_file_name=config_file_name)
    df = load_train_from_cfg(cfg=cfg)  # noqa pylint: disable=unused-variable

    pass


if __name__ == "__main__":
    main()

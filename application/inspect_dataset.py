from psycopt2d.load import load_train_from_cfg
from psycopt2d.utils.config_schemas import load_cfg_as_pydantic


def main():
    config_file_name = "default_config.yaml"

    cfg = load_cfg_as_pydantic(config_file_name=config_file_name)
    df = load_train_from_cfg(cfg=cfg)

    pass


if __name__ == "__main__":
    main()

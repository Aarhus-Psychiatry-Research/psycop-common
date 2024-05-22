
from psycop.common.model_training.config_schemas.conf_utils import load_app_cfg_as_pydantic
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema


def setup(config_file_name: str, application_config_dir_relative_path: str) -> FullConfigSchema:
    """Setup the requirements to run the model training pipeline."""
    cfg = load_app_cfg_as_pydantic(
        config_file_name=config_file_name, config_dir_path_rel=application_config_dir_relative_path
    )

    return cfg

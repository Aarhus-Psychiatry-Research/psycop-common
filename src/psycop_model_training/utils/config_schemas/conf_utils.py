from typing import Optional, Any, Union

from psycop_model_training.utils.config_schemas.full_config import FullConfigSchema


def convert_omegaconf_to_pydantic_object(
    conf: DictConfig,
    allow_mutation: bool = False,
) -> FullConfigSchema:
    """Converts an omegaconf DictConfig to a pydantic object.

    Args:
        conf (DictConfig): Omegaconf DictConfig
        allow_mutation (bool, optional): Whether to make the pydantic object mutable. Defaults to False.
    Returns:
        FullConfig: Pydantic object
    """
    conf = OmegaConf.to_container(conf, resolve=True)  # type: ignore
    return FullConfigSchema(**conf, allow_mutation=allow_mutation)


def load_test_cfg_as_omegaconf(
    config_file_name: str,
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """Load config as omegaconf object."""
    with initialize(version_base=None, config_path="../../../tests/config/"):
        if overrides:
            cfg = compose(
                config_name=config_file_name,
                overrides=overrides,
            )
        else:
            cfg = compose(
                config_name=config_file_name,
            )

        # Override the type so we can get autocomplete and renaming
        # correctly working
        cfg: FullConfigSchema = cfg  # type: ignore

        gpu = cfg.project.gpu

        if not gpu and cfg.model.name == "xgboost":
            cfg.model.args["tree_method"] = "auto"

        return cfg


def load_test_cfg_as_pydantic(
    config_file_name,
    allow_mutation: bool = False,
    overrides: Optional[list[str]] = None,
) -> FullConfigSchema:
    """Load config as pydantic object."""
    cfg = load_test_cfg_as_omegaconf(
        config_file_name=config_file_name, overrides=overrides
    )

    return convert_omegaconf_to_pydantic_object(conf=cfg, allow_mutation=allow_mutation)


class BaseModel(PydanticBaseModel):
    """."""

    class Config:
        """An pydantic basemodel, which doesn't allow attributes that are not
        defined in the class."""

        allow_mutation = False
        arbitrary_types_allowed = True
        extra = Extra.forbid

    def __transform_attributes_with_str_to_object(
        self,
        output_object: Any,
        input_string: str = "str",
    ):
        for key, value in self.__dict__.items():
            if isinstance(value, str):
                if value.lower() == input_string.lower():
                    self.__dict__[key] = output_object

    def __init__(
        self,
        allow_mutation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.Config.allow_mutation = allow_mutation

        self.__transform_attributes_with_str_to_object(
            input_string="null",
            output_object=None,
        )
        self.__transform_attributes_with_str_to_object(
            input_string="false",
            output_object=False,
        )
        self.__transform_attributes_with_str_to_object(
            input_string="true",
            output_object=True,
        )


class WatcherSchema(BaseModel):
    """Configuration for watchers."""

    archive_all: bool
    keep_alive_after_training_minutes: Union[int, float]
    n_runs_before_eval: int
    verbose: bool

"""Utilities for converting config yamls to pydantic objects. 

Helpful because it makes them:
- Addressable with intellisense,
- Refactorable with IDEs, 
- Easier to document with docstrings and 
- Type checkable
"""

import pydantic
from hydra import compose, initialize
from omegaconf import DictConfig


def omegaconf_to_pydantic_cfg(cfg: DictConfig) -> pydantic.BaseModel:
    """Convert OmegaConf to pydantic config."""
    return pydantic.parse_obj_as(pydantic.BaseModel, cfg)


def main():
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name="defualt_config.yaml",
        )

    pydantic_obj = omegaconf_to_pydantic_cfg(cfg)

    pass


if __name__ == "__main__":
    main()

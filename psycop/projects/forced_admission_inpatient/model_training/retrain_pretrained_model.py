"""Script using the train_model module to train a model.

Required to allow the trainer_spawner to point towards a python script
file, rather than an installed module.
"""
import faulthandler
import json
import sys
from pathlib import Path
from typing import Union

from omegaconf import DictConfig

from psycop.common.model_training.application_modules.train_model.main import train_model
from psycop.common.model_training.config_schemas.conf_utils import (
    convert_omegaconf_to_pydantic_object,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_CFG_PATH = Path(
    "E:/shared_resources/forced_admissions_inpatient/models/only_text_with_sentence_transformer_and_tfidf_750_embeddings/pipeline_eval/puncturation-unbeset/hamosemassecuite/cfg.json"
)


def _get_cfg_from_json_file(path: Path) -> FullConfigSchema:
    # Loading the json instead of the .pkl makes us independent
    # of whether the imports in psycop-common model-training have changed
    # TODO: Note that this means assigning to the cfg property does nothing, since it's recomputed every time it's called
    return FullConfigSchema.parse_obj(json.loads(json.loads(path.read_text())))


def main(cfg: Union[DictConfig, FullConfigSchema]) -> float:
    """Main."""
    if not isinstance(cfg, FullConfigSchema):
        cfg = convert_omegaconf_to_pydantic_object(cfg)

    if sys.platform == "win32":
        (PROJECT_ROOT / "wandb" / "debug-cli.onerm").mkdir(exist_ok=True, parents=True)

    return train_model(cfg=cfg)


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    cfg = _get_cfg_from_json_file(path=PRETRAINED_CFG_PATH)
    main(cfg)

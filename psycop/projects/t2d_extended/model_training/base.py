from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry

if __name__ == "__main__":
    populate_baseline_registry()

    train_baseline_model_from_cfg(
        cfg=PsycopConfig().from_disk(Path(__file__).parent / "t2d_base.cfg")
    )

from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import (
    train_baseline_model,
)
from psycop.common.model_training_v2.config.config_utils import (
    load_baseline_config,
)

if __name__ == "__main__":
    config = load_baseline_config(Path(__file__).parent / "cvd_baseline.cfg")
    train_baseline_model(config)


from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import (
    train_baseline_model,
)
from psycop.common.model_training_v2.config.config_utils import (
    load_baseline_config,
)
from psycop.projects.cvd.model_training.populate_cvd_registry import (
    populate_with_cvd_registry,
)

if __name__ == "__main__":
    populate_with_cvd_registry()
    config = load_baseline_config(Path(__file__).parent / "cvd_baseline.cfg")
    train_baseline_model(config)

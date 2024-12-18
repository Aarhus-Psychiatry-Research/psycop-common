from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.AcuteSomaticAdmission.model_training.populate_somatic_registry import (
    populate_with_somatic_registry,
)

if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_somatic_registry()
    train_baseline_model(Path(__file__).parent / "somatic_crossval.cfg")

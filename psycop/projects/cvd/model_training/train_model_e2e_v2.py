from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model
from psycop.projects.cvd.model_training.populate_cvd_registry import populate_with_cvd_registry

if __name__ == "__main__":
    populate_with_cvd_registry()
    train_baseline_model(Path(__file__).parent / "cvd_baseline.cfg")

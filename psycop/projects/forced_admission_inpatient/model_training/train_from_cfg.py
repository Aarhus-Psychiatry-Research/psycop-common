from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry

if __name__ == "__main__":
    populate_baseline_registry()
    train_baseline_model(
        Path(
            "E:/shared_resources/forced_admissions_inpatient/models/full_model_with_text_features_train_val/pipeline_eval/chuddahs-caterwauls-eval-on-test/abrasiometerintergradient-eval-on-test//config.cfg"
        )
    )

from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry

if __name__ == "__main__":
    populate_baseline_registry()
    #train_baseline_model(Path(__file__).parent / "text_exp_combined_tfidf_encoder_config.cfg")
    train_baseline_model(Path(__file__).parent / "text_exp_keywords.cfg")
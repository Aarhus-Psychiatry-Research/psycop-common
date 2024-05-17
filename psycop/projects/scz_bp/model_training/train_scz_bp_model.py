from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import populate_scz_bp_registry

if __name__ == "__main__":
    populate_baseline_registry()
    populate_scz_bp_registry()  # noqa: ERA001
    # train_baseline_model(Path(__file__).parent / "config" / "individual_outcomes" / "bp_structured_text_xgboost_ddpm.cfg")
    train_baseline_model(
        Path(__file__).parent
        / "config"
        / "individual_outcomes"
        / "scz_structured_text_xgboost_ddpm.cfg"
    )

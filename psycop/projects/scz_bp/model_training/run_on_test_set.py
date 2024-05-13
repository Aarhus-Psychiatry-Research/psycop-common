from pathlib import Path


from joblib import Parallel, delayed
from psycop.common.model_training_v2.config.baseline_pipeline import (
    train_baseline_model,
)
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)
from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import (
    populate_scz_bp_registry,
)


def fit_with_populated_registries(cfg_file: Path) -> float:
    populate_baseline_registry()
    populate_scz_bp_registry()
    result = train_baseline_model(cfg_file=cfg_file)
    return result

if __name__ == "__main__":
    
    cfg_paths = list((Path(__file__).parent / "config" / "test_set").glob("*.cfg"))
    Parallel(n_jobs=4)(delayed(fit_with_populated_registries)(cfg_file=cfg_path) for cfg_path in cfg_paths)

from pathlib import Path

from confection import Config
import polars as pl

# Set path to BaselineSchema for the run
## Load dataset with predictions after training
## Load validation dataset
## 1:1 join metadata cols to predictions
# Plot performance by
## Age
## Sex
## Calendar time
## Diagnosis type (scz or bp)
## Time to event
## Time from first visit
# Table of performance (sens, spec, ppv, f1) by threshold
## Confusion matrix at specified threshold
# Plot feature importance
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema

cfg_path = OVARTACI_SHARED_DIR / "scz_bp" / "experiments" / "l1"

def load_and_resolve_cfg(path: Path) -> BaselineSchema:
    cfg = Config().from_disk(path)
    cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg))
    return cfg_schema



if __name__ == "__main__":
    cfg = load_and_resolve_cfg(path=cfg_path / "config.json")

    pred_df = pl.read_parquet(cfg.project_info.experiment_path / "eval_df.parquet")


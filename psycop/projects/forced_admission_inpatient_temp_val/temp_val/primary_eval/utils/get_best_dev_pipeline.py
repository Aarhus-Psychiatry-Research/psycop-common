import pickle
from pathlib import Path
from typing import cast

from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import RunGroup
from psycop.projects.forced_admission_inpatient_temp_val.temp_val.primary_eval.utils.model_configs import (
    MODELS,
)


def get_best_model_cfg(model_name: str, model_algorithm: int) -> PsycopConfig:
    """
    Returns the loaded cfg.pkl for the best run of a model.
    """

    model_cfg = MODELS[model_name]

    group = RunGroup(model_name=model_name, group_name=cast(str, model_cfg["group_name"]))

    run_name = group.get_best_runs_by_lookahead()[model_algorithm, 2]

    run_dir = group.group_dir / run_name

    cfg_path = run_dir / "cfg.pkl"

    if not cfg_path.exists():
        raise FileNotFoundError(f"cfg.pkl not found at: {cfg_path}")

    # 4. Load config
    with Path.open(cfg_path, "rb") as f:
        cfg = pickle.load(f)

    if hasattr(cfg, "model_dump"):
        cfg = cfg.model_dump()

    cfg = PsycopConfig(cfg)

    return cfg


if __name__ == "__main__":
    cfg = get_best_model_cfg(model_name="full_model_without_text_features", model_algorithm=1)

    print(cfg)

import copy
from pathlib import Path

from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.cvd.model_training.hyperparam import hyperparameter_search
from psycop.projects.cvd.model_training.layers import train_cvd_layers
from psycop.projects.cvd.model_training.logistic_regression import logistic_regression_hyperparam
from psycop.projects.cvd.model_training.lookbehinds import train_with_lookbehinds
from psycop.projects.cvd.model_training.populate_cvd_registry import populate_with_cvd_registry
from psycop.projects.cvd.model_training.score2 import train_with_score2

if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_cvd_registry()

    cfg = PsycopConfig().from_disk(Path(__file__).parent / "ect_baseline.cfg")

    train_with_score2(cfg=copy.deepcopy(cfg))
    train_cvd_layers(cfg=copy.deepcopy(cfg))
    hyperparameter_search(cfg=copy.deepcopy(cfg))
    train_with_lookbehinds(cfg=copy.deepcopy(cfg), lookbehinds=[90, 365, 730])
    logistic_regression_hyperparam(cfg=copy.deepcopy(cfg))

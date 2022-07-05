from argparse import Namespace
from typing import Dict

from hydra import compose, initialize
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from psycopt2d.train_model import (
    create_model,
    create_preprocessing_pipeline,
    train_with_synth_data,
)


def dict_to_namespace(dict: Dict):
    """Recursively convert a nested dictionary to a namespace.

    Args:
        dict (Dict): Source dictionary
    """
    ns = Namespace()
    for k, v in dict.items():
        if isinstance(v, Dict):
            v = dict_to_namespace(v)
        setattr(ns, k, v)
    return ns


def test_basic_pipeline():
    with initialize(version_base=None, config_path="./configs"):
        cfg = compose(config_name="test_basic_pipeline.yaml")

        OUTCOME_COL_NAME = (
            f"outc_dichotomous_t2d_within_{cfg.data.lookahead_days}_days_max_fallback_0"
        )

        preprocessing_pipe = create_preprocessing_pipeline(cfg)
        mdl = create_model(cfg)

        if len(preprocessing_pipe.steps) != 0:
            pipe = Pipeline([("preprocessing", preprocessing_pipe), ("mdl", mdl)])
        else:
            pipe = Pipeline([("mdl", mdl)])

        OUTCOME_COL_NAME = "outc_dichotomous_t2d_within_30_days_max_fallback_0"
        X_eval, y, y_hat_prob = train_with_synth_data(cfg, OUTCOME_COL_NAME, pipe)

        assert round(roc_auc_score(y, y_hat_prob), 3) == 0.885

import copy
import re
from pathlib import Path
from typing import Any, Literal

import optuna
from confection import Config
from optuna import Study, Trial

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.hyperparameter_suggester.hyperparameter_suggester import (
    SuggesterSpace,
    suggest_hyperparams_from_cfg,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)

from ..config.baseline_pipeline import train_baseline_model_from_schema


class OptunaHyperParameterOptimization:
    @staticmethod
    def _validate_suggester_in_configspace(cfg: dict[str, Any]) -> None:
        has_suggester = OptunaHyperParameterOptimization._check_if_suggester_in_configspace(cfg=cfg)
        if not has_suggester:
            raise ValueError(
                "No suggesters found in config. Add one to conduct hyperparameter optimization."
            )

    @staticmethod
    def _check_if_suggester_in_configspace(cfg: dict[str, Any]) -> bool:
        """Recursively checks if any value in the dictionary or its nested dictionaries
        is of type SuggesterSpace or Suggester."""
        for _, value in cfg.items():
            if isinstance(value, (SuggesterSpace, Suggester)):
                return True
            if isinstance(
                value, dict
            ) and OptunaHyperParameterOptimization._check_if_suggester_in_configspace(value):
                return True
        return False

    @staticmethod
    def _check_if_any_key_matches_regex(regex_string: str, dictionary: dict[str, Any]) -> bool:
        """Check if any of the keys in the dictionary match the regex."""
        return any(re.match(regex_string, key) for key in dictionary)

    @staticmethod
    def _resolve_only_registries_matching_regex(
        cfg: dict[str, Any], regex_string: str
    ) -> dict[str, Any]:
        """Resolves only suggesters in a nested config. Suggesters are identified
        by being registered in with 'suggester' in the registry name"""
        cfg_copy = copy.deepcopy(cfg)

        for key, value in cfg_copy.items():
            if isinstance(value, dict):
                # check if match regex for suggester)
                if OptunaHyperParameterOptimization._check_if_any_key_matches_regex(
                    regex_string=regex_string, dictionary=value
                ):
                    cfg_copy[key] = BaselineRegistry.resolve({key: value})[key]
                else:
                    cfg_copy[
                        key
                    ] = OptunaHyperParameterOptimization()._resolve_only_registries_matching_regex(
                        cfg=value, regex_string=regex_string
                    )
        return cfg_copy

    @staticmethod
    def _optuna_objective(trial: Trial, cfg_with_resolved_suggesters: dict[str, Any]) -> float:
        concrete_config = suggest_hyperparams_from_cfg(
            base_cfg=cfg_with_resolved_suggesters, trial=trial
        )
        concrete_config_schema = BaselineSchema(**BaselineRegistry.resolve(concrete_config))
        return train_baseline_model_from_schema(cfg=concrete_config_schema)

    @staticmethod
    def conduct_hyperparameter_optimization_from_file(
        cfg_file: Path, n_trials: int, n_jobs: int, direction: Literal["maximize", "minimize"]
    ) -> Study:
        cfg = Config().from_disk(cfg_file)

        cfg_with_resolved_suggesters = (
            OptunaHyperParameterOptimization()._resolve_only_registries_matching_regex(
                cfg=cfg, regex_string="^@.*suggesters$"
            )
        )
        OptunaHyperParameterOptimization._validate_suggester_in_configspace(
            cfg=cfg_with_resolved_suggesters
        )

        study = optuna.create_study(direction=direction)
        study.optimize(
            lambda trial: OptunaHyperParameterOptimization()._optuna_objective(
                trial=trial, cfg_with_resolved_suggesters=cfg_with_resolved_suggesters
            ),
            n_trials=n_trials,
            n_jobs=n_jobs,
        )
        return study

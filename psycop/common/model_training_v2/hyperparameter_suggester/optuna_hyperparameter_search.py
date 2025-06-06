import copy
import re
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Literal

import joblib
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
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric

from ..config.populate_registry import populate_baseline_registry


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
                    cfg_copy[key] = (
                        OptunaHyperParameterOptimization()._resolve_only_registries_matching_regex(cfg=value, regex_string=regex_string)
                    )
        return cfg_copy

    @staticmethod
    def _optuna_objective(
        trial: Trial,
        cfg_with_resolved_suggesters: dict[str, Any],
        custom_populate_registry_fn: None | Callable[[], None],
    ) -> float:
        concrete_config = suggest_hyperparams_from_cfg(
            base_cfg=cfg_with_resolved_suggesters, trial=trial
        )

        populate_baseline_registry()
        if custom_populate_registry_fn:
            custom_populate_registry_fn()

        concrete_config_schema = BaselineSchema(**BaselineRegistry.resolve(concrete_config))
        concrete_config_schema.logger.log_config(Config(concrete_config))
        concrete_config_schema.logger.log_metric(
            CalculatedMetric(name="trial_number", value=trial.number)
        )

        try:
            run_result = concrete_config_schema.trainer.train()
        except Exception as e:
            if "Input X contains NaN" in str(e):
                raise optuna.TrialPruned from e
            concrete_config_schema.logger.fail(traceback.format_exc())
            raise
        return run_result.metric.value

    @staticmethod
    def _optimize_study(
        direction: Literal["maximize", "minimize"],
        study_name: str,
        n_trials: int,
        catch: tuple[type[Exception]],
        cfg_with_resolved_suggesters: dict[str, Any],
        custom_populate_registry_fn: None | Callable[[], None],
    ) -> Study:
        study = optuna.create_study(
            direction=direction,
            load_if_exists=True,
            study_name=study_name,
            storage=f"sqlite:///./{study_name}.db",
        )

        study.optimize(
            lambda trial: OptunaHyperParameterOptimization()._optuna_objective(
                trial=trial,
                cfg_with_resolved_suggesters=cfg_with_resolved_suggesters,
                custom_populate_registry_fn=custom_populate_registry_fn,
            ),
            n_trials=n_trials,
            catch=catch,
        )
        return study

    @staticmethod
    def from_cfg(
        cfg: Config,
        n_trials: int,
        n_jobs: int,
        study_name: str,
        direction: Literal["maximize", "minimize"],
        catch: tuple[type[Exception]],
        custom_populate_registry_fn: None | Callable[[], None],
    ) -> Sequence[Study]:
        cfg_with_resolved_suggesters = (
            OptunaHyperParameterOptimization()._resolve_only_registries_matching_regex(
                cfg=cfg, regex_string="^@.*suggesters$"
            )
        )
        OptunaHyperParameterOptimization._validate_suggester_in_configspace(
            cfg=cfg_with_resolved_suggesters
        )

        # instantiate the study in case it does not already exist
        study_ = optuna.create_study(  # noqa
            direction=direction,
            load_if_exists=True,
            study_name=study_name,
            storage=f"sqlite:///./{study_name}.db",
        )

        studies = joblib.Parallel(n_jobs)(
            joblib.delayed(OptunaHyperParameterOptimization._optimize_study)(
                direction=direction,
                n_trials=n_trials // n_jobs,
                study_name=study_name,
                catch=catch,
                cfg_with_resolved_suggesters=cfg_with_resolved_suggesters,
                custom_populate_registry_fn=custom_populate_registry_fn,
            )
            for _ in range(n_jobs)
        )

        return studies  # type: ignore

    @staticmethod
    def from_file(
        cfg_file: Path,
        n_trials: int,
        n_jobs: int,
        study_name: str,
        direction: Literal["maximize", "minimize"],
        catch: tuple[type[Exception]],
        custom_populate_registry_fn: None | Callable[[], None] = None,
    ) -> Sequence[Study]:
        cfg = Config().from_disk(cfg_file)
        studies = OptunaHyperParameterOptimization.from_cfg(
            n_trials=n_trials,
            n_jobs=n_jobs,
            study_name=study_name,
            direction=direction,
            catch=catch,
            cfg=cfg,
            custom_populate_registry_fn=custom_populate_registry_fn,
        )
        return studies

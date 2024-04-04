from typing import Any, Literal

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from optuna import Trial
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.suggester_spaces import (
    CategoricalSpace,
    CategoricalSpaceT,
    FloatSpace,
    FloatSpaceT,
    IntegerSpace,
    IntegerspaceT,
    SingleValue,
)
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep


class SyntheticDataAugmentation(BaseSampler):
    """Augment data with synthetic data generated from a model using synthcity.

    Args:
        model_name: Name of the GAN model to use.
        model_params: Parameters to pass to the model such n_iter, batch_size
        sampling_strategy: Strategy to use for sampling. Either 'minority' or 'all'.
            'minority' will only sample from the '1' class, 'all' will sample from both classes
            with same distribution as the training data. The name and values of the
            argument are required by imblearn.
        prop_augmented: Proportion of data to augment.
    """

    def __init__(
        self,
        model_name: str,
        model_params: dict[str, Any] | None = None,
        sampling_strategy: Literal["minority", "all"] = "all",
        prop_augmented: float = 0.5,
    ):
        if model_params is None:
            model_params = {}
        if model_name == "ddpm":
            model_params["is_classification"] = True  # required for conditional sampling
        self.model_name = model_name
        self.sampling_strategy = sampling_strategy
        self.prop_augmented = prop_augmented
        self.model_params = model_params
        self._sampling_type = "ensemble"  # imblearn internal

    def _fit_resample(  # type: ignore
        self,
        X: np.ndarray,  # type: ignore
        y: np.ndarray,  # type: ignore
    ) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
        """Fit the model to the data, generate synthetic data and combine with the original data."""
        outcome_col_name = "target"

        X_copy = pd.DataFrame(X.copy())
        y_copy = pd.Series(y.copy(), name=outcome_col_name)

        data_loader = GenericDataLoader(
            data=pd.concat([X_copy, y_copy], axis=1), target_column=outcome_col_name
        )

        model = Plugins().get(self.model_name, **self.model_params)
        model.fit(data_loader)

        # ratio_augmented  is the proportion of data to augment
        n_samples_to_generate = int(len(X_copy) * self.prop_augmented)

        match self.sampling_strategy:
            case "minority":
                # assuming the minority class (the class to upsample) is the 1 class
                cond = np.ones(n_samples_to_generate)
            case "all":
                cond = None
            case _:
                raise ValueError(
                    f"Invalid sampling strategy: {self.sampling_strategy}"
                    + "Must be one of ['minority', 'all']"
                )

        generated_data = model.generate(count=n_samples_to_generate, cond=cond).dataframe()

        X_gen = generated_data.drop(columns=outcome_col_name)
        X_gen.columns = X_copy.columns

        Xt = pd.concat([X_copy, X_gen], axis=0)
        yt = pd.concat([y_copy, generated_data[outcome_col_name]], axis=0)

        return Xt.to_numpy(), yt.to_numpy()


@BaselineRegistry.estimator_steps.register("synthetic_data_augmentation")
def synthetic_data_augmentation_step(
    model_name: str,
    model_params: dict[str, Any] | None,
    sampling_strategy: Literal["minority", "all"],
    prop_augmented: float,
) -> ModelStep:
    return (
        "synthetic_data_augmentation",
        SyntheticDataAugmentation(
            model_name=model_name,
            model_params=model_params,
            sampling_strategy=sampling_strategy,
            prop_augmented=prop_augmented,
        ),
    )


@BaselineRegistry.estimator_steps_suggesters.register("synthetic_data_augmentation_suggester")
class SynthcityAugmentationSuggester(Suggester):
    def __init__(
        self,
        model_name: CategoricalSpaceT | str = "ddpm",
        lr: FloatSpaceT | float = (1e-5, 1e-1, True),
        batch_size: IntegerspaceT | int = 10000,
        num_timesteps: IntegerspaceT | int = 1000,
        n_iter: IntegerspaceT | int = 1000,
        sampling_strategy: CategoricalSpaceT | str = ("all", "minority"),
        prop_augmented: CategoricalSpaceT | float = (0.05, 0.1, 0.5, 1.0),
    ):
        self.model_name = (
            CategoricalSpace(choices=model_name)
            if not isinstance(model_name, str)
            else SingleValue(model_name)
        )
        self.lr = (
            FloatSpace.from_list_or_mapping(lr)
            if not isinstance(lr, (float, int))
            else SingleValue(lr)
        )
        self.batch_size = (
            IntegerSpace.from_list_or_mapping(batch_size)
            if not isinstance(batch_size, int)
            else SingleValue(batch_size)
        )
        self.num_timesteps = (
            IntegerSpace.from_list_or_mapping(num_timesteps)
            if not isinstance(num_timesteps, int)
            else SingleValue(num_timesteps)
        )
        self.n_iter = (
            IntegerSpace.from_list_or_mapping(n_iter)
            if not isinstance(n_iter, int)
            else SingleValue(n_iter)
        )
        self.sampling_strategy = (
            CategoricalSpace(choices=sampling_strategy)
            if not isinstance(sampling_strategy, str)
            else SingleValue(sampling_strategy)
        )
        self.prop_augmented = (
            CategoricalSpace(choices=prop_augmented)
            if not isinstance(prop_augmented, (float, int))
            else SingleValue(prop_augmented)
        )

    def suggest_hyperparameters(self, trial: Trial) -> dict[str, Any]:
        return {
            "@estimator_steps": "synthetic_data_augmentation",
            "model_name": self.model_name.suggest(trial, name="model_name"),
            "sampling_strategy": self.sampling_strategy.suggest(trial, name="sampling_strategy"),
            "prop_augmented": self.prop_augmented.suggest(trial, name="prop_augmented"),
            "model_params": {
                "lr": self.lr.suggest(trial, name="lr_tabddpm"),
                "batch_size": self.batch_size.suggest(trial, name="batch_size"),
                "num_timesteps": self.num_timesteps.suggest(trial, name="num_timesteps"),
                "n_iter": self.n_iter.suggest(trial, name="n_iter"),
            },
        }

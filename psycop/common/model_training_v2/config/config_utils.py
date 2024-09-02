import copy
import operator
from functools import reduce
from pathlib import Path
from typing import Any

import confection
from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry

populate_baseline_registry()


def resolve_and_fill_config(config_path: Path, fill_cfg_with_defaults: bool) -> dict[str, Any]:
    cfg = Config().from_disk(config_path)

    # Fill with defaults
    if fill_cfg_with_defaults:
        filled = BaselineRegistry.fill(cfg, validate=False)
        resolved = BaselineRegistry.resolve(filled)
        # Writing to disk happens after resolution, to ensure that the
        # config is valid
        if cfg != filled:
            filled.to_disk(config_path)
    else:
        resolved = BaselineRegistry.resolve(cfg)

    return resolved


def load_baseline_config(config_path: Path) -> BaselineSchema:
    """Loads the baseline config from disk and resolves it."""
    resolved = resolve_and_fill_config(config_path, fill_cfg_with_defaults=True)
    return BaselineSchema(**resolved)


class PsycopConfig(confection.Config):
    def retrieve(self, location: str) -> Any:
        """Get a value from the config.

        Args:
            location: The location of the value to get. E.g. "trainer.training_data.paths.0"

        Returns:
            Any: The value at the location.
        """
        visited = []
        path_parts = location.split(".")

        current = self
        for part in path_parts:
            current = current.get(part)
            if current is None:
                raise AttributeError(
                    f"At {'.'.join(visited)}, could not find {part}. \n\tTarget: {location}. \n\tCurrent config: {self}"
                )
            visited.append(part)

        return current

    def mut(self, location: str, value: Any) -> "PsycopConfig":
        """Mutate a value in the config. Errors if the value does not exist, ensuring no silent failures.

        Args:
            location: The location of the value to set. E.g. "trainer.training_data.paths.0"
            value: The value to set.
        """
        # Get the value at the location to check that it exists
        self.retrieve(location)

        # Set the value at the location
        *path, last = location.split(".")
        reduce(operator.getitem, path, self)[last] = value
        return self

    def add(self, location: str, value: Any) -> "PsycopConfig":
        """Add a value to the config. Can add values multiple layers deep.

        Args:
            location: The location of the value to add. E.g. "trainer.training_data.paths.0"
            value: The value to add.
        """
        if isinstance(
            value, dict
        ):  # Avoid having the same mutable dict in multiple locations in the tree
            value = copy.deepcopy(value)

        *path, last = location.split(".")

        # Go through each layer. If it does not exist, create it as an empty dict.
        cur = self
        for layer in path:
            try:
                cur = cur[layer]
            except KeyError:
                cur[layer] = {}
                cur = cur[layer]

        # Add the value at the location
        try:
            cur[last] = value
        except KeyError as e:
            raise ValueError(f"At {location}, unable to add value.") from e

        return self

    def rem(self, location: str) -> "PsycopConfig":
        """Remove a value from the config. Errors if the value does not exist, ensuring no silent failures.

        Args:
            location: The location of the value to remove. E.g. "trainer.training_data.paths.0"
        """
        *path, second_to_last, last = location.split(".")

        # Remove the value at the location
        reduce(operator.getitem, path, self)[second_to_last].pop(last)
        return self

    def from_disk(
        self,
        path: Path | str,
        *,
        interpolate: bool = True,
        overrides: dict[str, Any] = {},  # noqa: B006
    ) -> "PsycopConfig":
        """Load a config from a file.

        Args:
            path: The path to the file.
            interpolate: Whether to interpolate variables in the config.
            overrides: A dictionary of overrides to apply to the config.

        Returns:
            PsycopConfig: The loaded config.
        """
        return PsycopConfig(Config().from_disk(path, interpolate=interpolate, overrides=overrides))

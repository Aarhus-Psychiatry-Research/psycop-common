from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from confection import Config, ConfigValidationError

from psycop.common.model_training_v2.config.baseline_registry import (
    BaselineRegistry,
    RegistryWithDict,
)
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)

populate_baseline_registry()

STATIC_REGISTRY_CONFIG_DIR = Path(__file__).parent / "static_registry_configs"


def _convert_tuples_to_lists(d: dict[str, Any]) -> dict[str, Any]:
    for key, value in d.items():
        if isinstance(value, tuple):
            d[key] = list(value)
        elif isinstance(value, dict):
            d[key] = _convert_tuples_to_lists(value)
    return d


@dataclass(frozen=True)
class RegisteredFunction:
    fn_name: str
    registry_name: str
    container_registry: RegistryWithDict

    def to_dot_path(self) -> str:
        return f"{self.registry_name}.{self.fn_name}"

    def get_cfg_dir(self, top_level_dir: Path) -> Path:
        return top_level_dir / self.registry_name / self.fn_name

    def has_example_cfg(self, example_top_dir: Path) -> bool:
        return len(list(self.get_cfg_dir(example_top_dir).glob("*.cfg"))) > 0

    def write_scaffolding_cfg(self, example_top_dir: Path) -> Path:
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg_dir = self.get_cfg_dir(example_top_dir)
        cfg_dir.mkdir(parents=True, exist_ok=True)

        example_path = (cfg_dir / f"{self.fn_name}_{current_datetime}.cfg").open("w")
        with example_path as f:
            f.write("[placeholder]")

        return example_path

    def get_example_cfgs(self, example_top_dir: Path) -> Sequence[Config]:
        cfgs = []
        for file in self.get_cfg_dir(example_top_dir).glob("*.cfg"):
            cfgs.append(Config().from_disk(file))

        return cfgs


def _identical_config_exists(
    filled_cfg: Config,
    fn: RegisteredFunction,
    top_level_dir: Path,
) -> bool:
    """Check if an identical config file already exists on disk."""
    files_with_same_function_name = fn.get_cfg_dir(top_level_dir=top_level_dir).glob(
        "*.cfg",
    )

    for file in files_with_same_function_name:
        file_cfg = Config().from_disk(file)
        # confection converts tuples to lists in the file config, so we need to convert
        # tuples to lists in the filled config to compare
        file_identical = _convert_tuples_to_lists(filled_cfg) == file_cfg
        if file_identical:
            return True
    return False


def _timestamped_cfg_to_disk(
    filled_cfg: Config,
    fn: RegisteredFunction,
    top_level_dir: Path,
) -> None:
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    filled_cfg.to_disk(
        fn.get_cfg_dir(top_level_dir=top_level_dir)
        / f"{fn.fn_name}_{current_datetime}.cfg",
    )


def get_registered_functions(
    container_registry: RegistryWithDict,
) -> Sequence[RegisteredFunction]:
    functions = []
    for registry_name, registry in container_registry.to_dict().items():
        for registered_function_name in registry.get_all():
            functions.append(
                RegisteredFunction(
                    registry_name=registry_name,
                    fn_name=registered_function_name,
                    container_registry=container_registry,
                ),
            )

    return functions


@dataclass(frozen=True)
class CouldNotGenerateConfigsError(Exception):
    fn: RegisteredFunction
    scaffolding_path: Path
    error: Exception


def generate_configs_from_registered_functions(
    registered_fns: Sequence[RegisteredFunction],
    example_cfg_dir: Path,
) -> bool:
    """Generate config files from registered functions in the baseline registry
    by filling with the default values. If no identical file exists on disk,
    write the config to disk. This is useful for testing whether old configs
    still work with the current codebase.
    Returns bool indicating whether any new configs were written to disk."""
    generated_new_configs = False
    config_validation_errors: Sequence[CouldNotGenerateConfigsError] = []

    for fn in registered_fns:
        cfg = Config(
            {"placeholder": {f"@{fn.registry_name}": f"{fn.fn_name}"}},
        )
        try:
            filled_cfg = fn.container_registry.fill(cfg)
        except ConfigValidationError as e:
            if fn.has_example_cfg(example_top_dir=example_cfg_dir):
                continue

            # Create an empty file at the location
            scaffolding_path = fn.write_scaffolding_cfg(example_top_dir=example_cfg_dir)

            config_validation_errors.append(
                CouldNotGenerateConfigsError(
                    fn=fn,
                    error=e,
                    scaffolding_path=scaffolding_path,
                ),
            )

            continue

        # If none, write to disk
        if not _identical_config_exists(
            filled_cfg=filled_cfg,
            top_level_dir=example_cfg_dir,
            fn=fn,
        ):
            generated_new_configs = True
            _timestamped_cfg_to_disk(
                filled_cfg=filled_cfg,
                fn=fn,
                top_level_dir=example_cfg_dir,
            )

    if config_validation_errors:
        fn_errors = "\n".join(
            f"{e.fn.to_dot_path()}: {e.scaffolding_path}"
            for e in config_validation_errors
        )

        raise Exception(
            f"""Encounted ConfigValidationErrors. For each, it means that either
    a) No example config exists at the cfg dir. In this case, scaffolding-configs have been generated for you.
    b) The function has changed in a way that breaks backwards compatability by e.g. adding a new, non-default argument.

Errors:
{fn_errors}
                """,
        )

    return generated_new_configs


if __name__ == "__main__":
    generate_configs_from_registered_functions(
        registered_fns=get_registered_functions(BaselineRegistry()),
        example_cfg_dir=STATIC_REGISTRY_CONFIG_DIR,
    )
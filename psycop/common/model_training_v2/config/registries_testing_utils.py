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


def _identical_config_exists(filled_cfg: Config, base_file_path: Path) -> bool:
    """Check if an identical config file already exists on disk."""
    files_with_same_function_name = base_file_path.parent.glob(f"{base_file_path.name}*")

    for file in files_with_same_function_name:
        file_cfg = Config().from_disk(file)
        # confection converts tuples to lists in the file config, so we need to convert
        # tuples to lists in the filled config to compare
        file_identical = _convert_tuples_to_lists(filled_cfg) == file_cfg
        if file_identical:
            return True
    return False


def _timestamped_cfg_to_disk(filled_cfg: Config, base_filename: Path) -> None:
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename.parent.mkdir(parents=True, exist_ok=True)
    out_filename = base_filename.name + f"-{current_datetime}.cfg"

    filled_cfg.to_disk(base_filename.parent / out_filename)


@dataclass(frozen=True)
class RegisteredFunction:
    registry_name: str
    container_registry: RegistryWithDict
    name: str

    def to_dot_path(self) -> str:
        return f"{self.registry_name}.{self.name}"

    def get_cfg_path(self, top_level_dir: Path) -> Path:
        return top_level_dir / self.registry_name / f"{self.name}.cfg"

    def has_example_cfg(self, example_top_dir: Path) -> bool:
        return self.get_cfg_path(example_top_dir).exists()

    def get_example_cfg(self, example_top_dir: Path) -> Config:
        return Config().from_disk(self.get_cfg_path(example_top_dir))

def get_registered_functions(
    container_registry: RegistryWithDict,
) -> Sequence[RegisteredFunction]:
    functions = []
    for registry_name, registry in container_registry.to_dict().items():
        for registered_function_name in registry.get_all():
            functions.append(
                RegisteredFunction(
                    registry_name=registry_name,
                    name=registered_function_name,
                    container_registry=container_registry,
                ),
            )

    return functions


def generate_configs_from_registered_functions(
    registered_fns: Sequence[RegisteredFunction],
    output_dir: Path,
) -> bool:
    """Generate config files from registered functions in the baseline registry
    by filling with the default values. If no identical file exists on disk,
    write the config to disk. This is useful for testing whether old configs
    still work with the current codebase.
    Returns bool indicating whether any new configs were written to disk."""
    generated_new_configs = False
    for fn in registered_fns:
        cfg = Config(
            {"placeholder": {f"@{fn.registry_name}": f"{fn.name}"}},
        )
        try:
            filled_cfg = fn.container_registry.fill(cfg)
        except ConfigValidationError as e:
            raise Exception(
                f"""Encounted ConfigValidationError in {fn.to_dot_path()}. This means that either\n\ta) the function has changed
    in a way that breaks backwards compatability by e.g. adding a new, non-default argument or \n\tb) No default config options exist at {output_dir}.""",
            ) from e

        base_filename = fn.get_cfg_path(output_dir)

        # If none, write to disk
        if not _identical_config_exists(
            filled_cfg,
            base_filename,
        ):
            generated_new_configs = True
            filled_cfg.to_disk(base_filename)

    return generated_new_configs

if __name__ == "__main__":
    generate_configs_from_registered_functions(
        registered_fns=get_registered_functions(BaselineRegistry()),
        output_dir=STATIC_REGISTRY_CONFIG_DIR,
    )

from datetime import datetime
from pathlib import Path
from typing import Any

from confection import Config, ConfigValidationError

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)

populate_baseline_registry()

STATIC_REGISTRY_CONFIG_DIR = Path(__file__).parent / "static_registry_configs"


REGISTERED_FUNCTION_WHITELIST = {
    "pipe_constructor",
    "mock_suggester",
    "baseline_preprocessing_pipeline",
    "crossval_trainer",
    "split_trainer",
    "binary_classification_pipeline",
    "binary_classification",
    "lookbehind_combination_col_filter",
    "minimal_test_data",
}


def convert_tuples_to_lists(d: dict[str, Any]) -> dict[str, Any]:
    for key, value in d.items():
        if isinstance(value, tuple):
            d[key] = list(value)
        elif isinstance(value, dict):
            d[key] = convert_tuples_to_lists(value)
    return d


def _identical_config_exists(filled_cfg: Config, base_filename: Path) -> bool:
    """Check if an identical config file already exists on disk."""
    files_with_same_function_name = base_filename.parent.glob(f"{base_filename.name}*")

    for file in files_with_same_function_name:
        file_cfg = Config().from_disk(file)
        # confection converts tuples to lists in the file config, so we need to convert
        # tuples to lists in the filled config to compare
        file_identical = convert_tuples_to_lists(filled_cfg) == file_cfg
        if file_identical:
            return True
    return False


def _save_cfg_to_disk(filled_cfg: Config, base_filename: Path) -> None:
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename.parent.mkdir(parents=True, exist_ok=True)
    out_filename = base_filename.name + f"-{current_datetime}.yaml"

    filled_cfg.to_disk(base_filename.parent / out_filename)


def generate_configs_from_registered_functions() -> bool:
    """Generate config files from registered functions in the baseline registry
    by filling with the default values. If no identical file exists on disk,
    write the config to disk. This is useful for testing whether old configs
    still work with the current codebase.
    Returns bool indicating whether any new configs were written to disk."""
    registries_dict = BaselineRegistry().to_dict()

    generated_new_configs = False
    for registry_name, registry in registries_dict.items():
        for registered_function_name in registry.get_all():
            cfg = Config(
                {"placeholder": {f"@{registry_name}": f"{registered_function_name}"}},
            )
            try:
                filled_cfg = BaselineRegistry().fill(cfg)
            except ConfigValidationError as e:
                if any(
                    whitelisted_function in str(e)
                    for whitelisted_function in REGISTERED_FUNCTION_WHITELIST
                ):
                    print(
                        f"Encountered ConfigValidationError in {registered_function_name}, but whitelisted, skipping",
                    )
                    continue
                if "*" in str(e):
                    raise Exception(
                        f"""Encountered ConfigValidationError in {registered_function_name}, with a function that's not whitelisted.
        This means that the function has changed in a way that breaks backwards compatability, by adding a
        *args argument. If this is intentional, add the function name to the whitelist in registries_testing_utils.py.
        Otherwise, create a copy of the function before the change and update the name of the new function in the registry,
        e.g. func_name_v2.""",
                    ) from e

                raise Exception(
                    f"""Encounted ConfigValidationError in {registered_function_name}. This means that the function has changed
        in a way that breaks backwards compatability by e.g. adding a new, non-default argument.
        Ensure that all arguments have default values and can be resolved by the registry.""",
                ) from e

            base_filename = (
                STATIC_REGISTRY_CONFIG_DIR / registry_name / registered_function_name
            )
            ## check for identical files on disk
            identical_file_exists = _identical_config_exists(
                filled_cfg,
                base_filename,
            )
            # if none, write to disk
            if not identical_file_exists:
                generated_new_configs = True
                _save_cfg_to_disk(filled_cfg, base_filename)
    return generated_new_configs


if __name__ == "__main__":
    generate_configs_from_registered_functions()

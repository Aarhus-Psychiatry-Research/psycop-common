from datetime import datetime
from pathlib import Path

from confection import Config, ConfigValidationError

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)

populate_baseline_registry()

STATIC_REGISTRY_CONFIG_DIR = Path(__file__).parent / "static_registry_configs"


def _check_if_identical_file_exists(filled_cfg: Config, base_filename: Path) -> bool:
    """Check if an identical config file already exists on disk."""
    files_with_same_function_name = base_filename.parent.glob(f"{base_filename.name}*")

    for file in files_with_same_function_name:
        file_cfg = Config().from_disk(file)
        file_identical = filled_cfg == file_cfg
        if file_identical:
            return True
    return False


def generate_configs_from_registered_functions():
    """Generate config files from registered functions in the baseline registry
    by filling with the default values. If no identical file exists on disk,
    write the config to disk. This is useful for testing whether old configs
    still work with the current codebase."""
    registries_dict = BaselineRegistry().to_dict()

    for registry_name, registry in registries_dict.items():
        for registered_function_name in registry.get_all():
            cfg = Config(
                {"placeholder": {f"@{registry_name}": f"{registered_function_name}"}},
            )
            try:
                filled_cfg = BaselineRegistry().fill(cfg)
            except ConfigValidationError as e:
                if "field required" in str(e):
                    print(
                        f"Encountered ConfigValidationError in {registered_function_name}, skipping",
                    )
                    # This error is raised for e.g. 'pipe_constructor' which takes
                    # *args as input and which we cannot resolve without knowing
                    # the args.
                    continue

            base_filename = (
                STATIC_REGISTRY_CONFIG_DIR / registry_name / registered_function_name
            )
            ## check for identical files on disk
            identical_file_exists = _check_if_identical_file_exists(
                filled_cfg,
                base_filename,
            )
            # if none, write to disk
            if not identical_file_exists:
                current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename.parent.mkdir(parents=True, exist_ok=True)
                out_filename = base_filename.name + f"-{current_datetime}.yaml"

                filled_cfg.to_disk(base_filename.parent / out_filename)

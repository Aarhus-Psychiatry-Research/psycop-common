import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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
class RegisteredCallable:
    callable_name: str
    registry_name: str
    container_registry: RegistryWithDict
    module: str

    def to_dot_path(self) -> str:
        return f"{self.registry_name}.{self.callable_name}"

    def get_cfg_dir(self, top_level_dir: Path) -> Path:
        return top_level_dir / self.registry_name / self.callable_name

    def has_example_cfg(self, example_top_dir: Path) -> bool:
        return len(list(self.get_cfg_dir(example_top_dir).glob("*.cfg"))) > 0

    def write_scaffolding_cfg(
        self,
        example_top_dir: Path,
    ) -> Path:
        cfg_dir = self.get_cfg_dir(example_top_dir)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        example_path = cfg_dir / f"{self.callable_name}_{current_datetime}.cfg"

        with example_path.open("w") as f:
            arg_names = self._get_callable_arg_names()
            f.write(
                f"""
# Example cfg for {self.callable_name}
# You can find args at:
#    {self.module}
[{self.callable_name}]
@{self.registry_name} = "{self.callable_name}"
"""
                + "\n".join(f"{arg_name} = " for arg_name in arg_names),
            )

        return example_path

    def _get_callable_arg_names(self) -> Sequence[str]:
        """Get the names of the arguments of the callable.

        If the callable is a class, get args of __init__, omitting self."""
        if inspect.isfunction(self.callable_obj):
            arg_names = inspect.getfullargspec(self.callable_obj).args
        elif inspect.isclass(self.callable_obj):
            arg_names = inspect.getfullargspec(self.callable_obj.__init__).args[1:]
            # Slice from 1 to omit self

            has_starargs = (
                inspect.getfullargspec(self.callable_obj.__init__).varargs is not None
            )
            if has_starargs:
                arg_names = [*arg_names, f"\n[{self.callable_name}.*]\nplaceholder_*"]
        else:
            raise ValueError(
                f"Expected {self.callable_name} to be a function or class",
            )

        return arg_names

    @property
    def callable_obj(self) -> Callable:  # type: ignore
        return self.container_registry.get(self.registry_name, self.callable_name)

    def get_example_cfgs(self, example_top_dir: Path) -> Sequence[Config]:
        cfgs = []
        for file in self.get_cfg_dir(example_top_dir).glob("*.cfg"):
            cfgs.append(Config().from_disk(file))

        return cfgs


def _identical_config_exists(
    filled_cfg: Config,
    fn: RegisteredCallable,
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


def _new_timestamped_cfg_to_disk(
    filled_cfg: Config,
    fn: RegisteredCallable,
    top_level_dir: Path,
) -> None:
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = (
        fn.get_cfg_dir(top_level_dir=top_level_dir)
        / f"{fn.callable_name}_{current_datetime}.cfg"
    )
    filepath.parent.mkdir(exist_ok=True, parents=True)

    filled_cfg.to_disk(filepath)

    # Prepend location to filepath
    with filepath.open("r") as f:
        contents = f.read()

    with filepath.open("w") as f:
        f.write(
            f"""# Example cfg for {fn.callable_name}
# You can find args at:
#    {fn.module}\n{contents}""",
        )


def get_registered_functions(
    container_registry: RegistryWithDict,
) -> Sequence[RegisteredCallable]:
    functions = []
    for registry_name, registry in container_registry.to_dict().items():
        for registered_function_name in registry.get_all():
            functions.append(
                RegisteredCallable(
                    registry_name=registry_name,
                    callable_name=registered_function_name,
                    container_registry=container_registry,
                    module=registry.get_all()[registered_function_name].__module__,
                ),
            )

    return functions


@dataclass(frozen=True)
class CouldNotGenerateConfigsError(Exception):
    fn: RegisteredCallable
    scaffolding_path: Path
    error: Exception


def generate_configs_from_registered_functions(
    registered_fns: Sequence[RegisteredCallable],
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
        placeholder_cfg = Config(
            {"placeholder": {f"@{fn.registry_name}": f"{fn.callable_name}"}},
        )
        try:
            # Fill with default values.
            filled_cfg = fn.container_registry.fill(placeholder_cfg)
        except ConfigValidationError as e:
            # Could not fill with default values.
            # This means we need to provide them in the example cfg.
            if fn.has_example_cfg(example_top_dir=example_cfg_dir):
                continue

            # Create a scaffolding cfg at the location
            scaffolding_path = fn.write_scaffolding_cfg(
                example_top_dir=example_cfg_dir,
            )

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
            _new_timestamped_cfg_to_disk(
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
    a) No example config exists at the cfg dir. In this case, scaffolding-configs have been generated for you, which can be filled in with a valid example config. These ensure that, when we edit the file in the future, we are alerted if the edits are not backwards compatible.
    b) The function has changed in a way that breaks backwards compatability by e.g. adding a new, non-default argument. In this case, you should create a new function with a new name in the (e.g. fn_v2), and register it in the registry. The old function should be archived in the archive folder.

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

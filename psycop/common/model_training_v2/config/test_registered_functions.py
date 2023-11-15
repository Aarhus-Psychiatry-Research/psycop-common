import os
from pathlib import Path

import pytest
from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import (
    BaselineRegistry,
    RegistryWithDict,
)
from psycop.common.model_training_v2.config.registries_testing_utils import (
    STATIC_REGISTRY_CONFIG_DIR,
    generate_configs_from_registered_functions,
    get_registered_functions,
)

REGISTERED_FUNCTION_ERROR_MSG = """
Some of the changes made in this PR break backwards-compatability.
This means some historical configs will not be resolvable. To work around this:

If you added an argument to a function without a default:
Create a copy of the function before the change and add it to the archive folder.
The changed version of the function should be registered under a new name in the registry,
e.g. func_name_v2
"""


@pytest.mark.parametrize(
    ("source_registry", "output_dir"),
    [
        (BaselineRegistry(), Path(__file__).parent / "historical_registry_configs"),
    ],
)
def test_registered_functions(source_registry: RegistryWithDict, output_dir: Path):
    registered_fns = get_registered_functions(source_registry)

    generate_configs_from_registered_functions(
        registered_fns=registered_fns,
        output_dir=output_dir,
    )

    for fn in registered_fns:
        assert fn.has_example_cfg(output_dir)
        cfgs = fn.get_example_cfgs(output_dir)

        for example_cfg in cfgs:
            try:
                fn.container_registry.resolve(example_cfg)
            except Exception as e:
                raise Exception(
                    f"Failed to resolve {fn.to_dot_path()}.\n{REGISTERED_FUNCTION_ERROR_MSG}",
                ) from e

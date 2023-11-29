from pathlib import Path

import pytest

from psycop.common.model_training_v2.config.baseline_registry import (
    BaselineRegistry,
    RegistryWithDict,
)
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)
from psycop.common.model_training_v2.config.registries_testing_utils import (
    generate_configs_from_registered_functions,
    get_example_cfgs,
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
def test_registered_callables_should_have_valid_example_cfgs(
    source_registry: RegistryWithDict,
    output_dir: Path,
):
    populate_baseline_registry()
    registered_fns = get_registered_functions(source_registry)

    generate_configs_from_registered_functions(
        registered_fns=registered_fns,
        example_cfg_dir=output_dir,
    )

    for fn in registered_fns:
        missing_example_cfgs: list[ValueError] = []
        if not fn.has_example_cfg(output_dir):
            missing_example_cfgs.append(
                ValueError(
                    f"{fn.to_dot_path()} does not have an example cfg at {fn.get_cfg_dir(output_dir)}",
                ),
            )
        if missing_example_cfgs:
            raise Exception(
                f"Encountered the following errors:\n{missing_example_cfgs}",
            )

    cfgs = get_example_cfgs(output_dir)

    for example_cfg in cfgs:
        try:
            BaselineRegistry.resolve(example_cfg)
        except Exception as e:
            raise Exception(
                f"Failed to resolve {example_cfg}.\n{REGISTERED_FUNCTION_ERROR_MSG}",
            ) from e


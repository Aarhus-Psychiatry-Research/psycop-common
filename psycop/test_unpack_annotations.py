import inspect
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from .unpack_annotations import get_pretty_type_str


@dataclass(frozen=True)
class MyType:
    ...


@dataclass(frozen=True)
class UnpackExpectation:
    arg_name: str
    expected_annotation: str


def fn(
    base_arg: str,  # noqa: ARG001
    contained_type: Sequence[MyType],  # noqa: ARG001
    custom_type: MyType,  # noqa: ARG001
    new_union: int | str,  # noqa: ARG001
    new_optional: str | None,  # noqa: ARG001
):
    pass


@pytest.mark.parametrize(
    ("example"),
    [
        UnpackExpectation("base_arg", "str"),
        UnpackExpectation("contained_type", "Sequence[MyType]"),
        UnpackExpectation("custom_type", "MyType"),
        UnpackExpectation("new_union", "int | str"),
        UnpackExpectation("new_optional", "str | None"),
    ],
)
def test_unpack_annotations(example: UnpackExpectation):
    fn_annotations = inspect.get_annotations(fn)
    assert (
        get_pretty_type_str(fn_annotations[example.arg_name])
        == example.expected_annotation
    )

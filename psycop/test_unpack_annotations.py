import inspect
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from .unpack_annotations import get_pretty_annotation_str


@dataclass(frozen=True)
class MyType:
    ...


@dataclass(frozen=True)
class UnpackTestExample:
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
        UnpackTestExample("base_arg", "str"),
        UnpackTestExample("contained_type", "Sequence[MyType]"),
        UnpackTestExample("custom_type", "MyType"),
        UnpackTestExample("new_union", "int | str"),
        UnpackTestExample("new_optional", "str | None"),
    ],
)
def test_unpack_annotations(example: UnpackTestExample):
    fn_annotations = inspect.get_annotations(fn)
    assert (
        get_pretty_annotation_str(fn_annotations[example.arg_name])
        == example.expected_annotation
    )

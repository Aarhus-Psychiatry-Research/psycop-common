from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from .unpack_annotations import get_pretty_type_str


@dataclass(frozen=True)
class MyType:
    ...


@dataclass(frozen=True)
class UnpackExpectation:
    type_annotation: type
    expected_annotation: str


@pytest.mark.parametrize(
    ("example"),
    [
        UnpackExpectation(str, "str"),
        UnpackExpectation(Sequence[MyType], "Sequence[MyType]"),
        UnpackExpectation(MyType, "MyType"),
        UnpackExpectation(int | str, "int | str"),
        UnpackExpectation(str | None, "str | None"),
    ],
)
def test_unpack_annotations(example: UnpackExpectation):
    assert get_pretty_type_str(example.type_annotation) == example.expected_annotation

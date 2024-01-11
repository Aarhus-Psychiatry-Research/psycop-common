import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional, Union

import pytest

from .test import get_pretty_annotation_str


@dataclass(frozen=True)
class MyType:
    ...


@dataclass(frozen=True)
class UnpackTestExample:
    arg_name: str
    annotation_str: str


def fn(
    base_arg: str,
    contained_type: Sequence[MyType],
    custom_type: MyType,
    new_union: int | str,
    new_optional: str | None,
):
    pass


@pytest.mark.parametrize(
    ("example"),
    [
        UnpackTestExample(arg_name="base_arg", annotation_str="str"),
        UnpackTestExample(arg_name="contained_type", annotation_str="Sequence[MyType]"),
        UnpackTestExample(arg_name="custom_type", annotation_str="MyType"),
        UnpackTestExample(arg_name="new_union", annotation_str="int | str"),
        UnpackTestExample(arg_name="new_optional", annotation_str="str | None"),
    ],
)
def test_unpack_annotations(example: UnpackTestExample):
    fn_annotations = inspect.get_annotations(fn)
    assert (
        get_pretty_annotation_str(fn_annotations[example.arg_name])
        == example.annotation_str
    )

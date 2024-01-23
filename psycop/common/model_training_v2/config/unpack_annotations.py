import types
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TypeWrapper:
    name: str
    prefix: str
    suffix: str
    separator: str

    def to_wrapped_annotation(self, wrapped_types: Sequence[str]) -> str:
        return (
            f"{self.name}{self.prefix}{self.separator.join(wrapped_types)}{self.suffix}"
        )


def get_wrapping_type(annotation: types.GenericAlias) -> TypeWrapper:
    """Get the name of an annotation, including how it wraps contained annotations."""
    if isinstance(annotation, types.UnionType):
        return TypeWrapper(name="", prefix="", suffix="", separator=" | ")
    return TypeWrapper(name=annotation.__name__, prefix="[", suffix="]", separator=", ")


def get_pretty_type_str(annotation: types.GenericAlias | type) -> str:
    """Recursively unpacks an annotation to a string representation."""
    try:
        contained_type_strings = [
            get_pretty_type_str(arg)
            for arg in annotation.__args__  # type: ignore
        ]
        wrapping_type = get_wrapping_type(annotation)  # type: ignore
        return wrapping_type.to_wrapped_annotation(contained_type_strings)
    # If it contains no args, it is not a wrapping type. Just return the type.
    except AttributeError:
        if annotation is types.NoneType:
            return "None"
        if any(isinstance(annotation, value) for value in [int, float]):
            return str(annotation)
        return annotation.__name__

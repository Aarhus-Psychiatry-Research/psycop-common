import inspect
import types
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AnnotationWrapper:
    name: str
    prefix: str
    suffix: str

    def to_wrapped_annotation(self, annotation: str) -> str:
        return f"{self.name}{self.prefix}{annotation}{self.suffix}"


def get_wrapper_annotation(annotation: types.GenericAlias) -> AnnotationWrapper:
    """Get the name of an annotation, including how it wraps contained annotations."""
    if isinstance(annotation, types.UnionType):
        return AnnotationWrapper(name="", prefix="", suffix="")
    return AnnotationWrapper(name=annotation.__name__, prefix="[", suffix="]")


def get_pretty_annotation_str(annotation: types.GenericAlias) -> str:
    """Recursively unpacks an annotation to a string representation."""
    try:
        argument_strings = [
            get_pretty_annotation_str(arg) for arg in annotation.__args__
        ]
        annotation_wrapper = get_wrapper_annotation(annotation)
        return annotation_wrapper.to_wrapped_annotation(" | ".join(argument_strings))
    except AttributeError:
        if annotation is types.NoneType:
            return "None"
        return annotation.__name__

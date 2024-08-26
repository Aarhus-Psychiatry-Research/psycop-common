import pytest

from psycop.common.model_training_v2.config.config_utils import PsycopConfig


def test_retrieve():
    cfg = PsycopConfig({"a": {"b": {"c": 1}}})
    assert cfg.retrieve("a.b.c") == 1


def test_retrieve_error_if_path_does_not_exist():
    cfg = PsycopConfig({"a": {"b": {"c": 1}}})
    with pytest.raises(
        AttributeError,
        match="At a.b, could not find d. \n\tTarget: a.b.d. \n\tCurrent config: {'a': {'b': {'c': 1}}}",
    ):
        cfg.retrieve("a.b.d")


def test_mutate():
    cfg = PsycopConfig({"a": {"b": {"c": 1}}})
    cfg.mut("a.b.c", 2)
    assert cfg["a"]["b"]["c"] == 2


def test_mutate_error_if_path_does_not_exist():
    cfg = PsycopConfig({"a": {"b": {"c": 1}}})
    with pytest.raises(
        AttributeError,
        match="At a.b, could not find d. \n\tTarget: a.b.d. \n\tCurrent config: {'a': {'b': {'c': 1}}}",
    ):
        cfg.mut("a.b.d", 2)


def test_remove():
    cfg = PsycopConfig({"a": {"b": {"c": 1}}})
    cfg.rem("a.b.c")

    assert cfg["a"] == {"b": {}}


def test_remove_errors_if_not_present():
    cfg = PsycopConfig({"a": {"b": {"c": 1}}})

    with pytest.raises(KeyError, match=".*d.*"):
        cfg.rem("a.b.d")


def test_add_multilayer():
    """Add a value multiple layers deep."""
    cfg = PsycopConfig({"a": {}})

    cfg.add("a.b.d", 2)

    assert cfg["a"]["b"]["d"] == 2

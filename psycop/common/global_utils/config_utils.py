from collections.abc import MutableMapping
from typing import Any, Iterable


def flatten_nested_dict(
    d: MutableMapping[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Recursively flatten an infinitely nested config. E.g. {"level1":

    {"level2": "level3": {"level4": 5}}}} becomes:

    {"level1.level2.level3.level4": 5}.

    Args:
        d: Dict to flatten.
        parent_key: The parent key for the current dict, e.g. "level1" for the
            first iteration. Defaults to "".
        sep: How to separate each level in the dict. Defaults to ".".

    Returns:
        Dict: The flattened dict.
    """

    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_nested_dict(d=v, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def replace_symbols_in_dict_keys(
    d: dict[str, Any], symbol2replacement: dict[str, str]
):
    new_dict = {}
    for key, value in d.items():
        for symbol, replacement in symbol2replacement.items():
            key = key.replace(symbol, replacement)
        new_dict[key] = value
    return new_dict

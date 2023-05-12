import pickle as pkl
from pathlib import Path
from typing import Any


def read_pickle(filepath: Path) -> Any:
    filepath.parent.mkdir(exist_ok=True, parents=True)

    with Path(filepath).open("rb") as f:
        return pkl.load(f)


def write_to_pickle(obj_to_pickle: Any, filepath: Path):
    filepath.parent.mkdir(exist_ok=True, parents=True)

    with Path(filepath).open("wb") as f:
        pkl.dump(obj_to_pickle, f)

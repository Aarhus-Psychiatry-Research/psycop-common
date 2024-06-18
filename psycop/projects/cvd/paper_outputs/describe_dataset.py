from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from psycop.projects.cvd.cohort_examination.table_one.facade import table_one


class CVDArtifact(Protocol):
    def __call__(self, output_dir: Path) -> None: ...


if __name__ == "__main__":
    artifacts: Sequence[CVDArtifact] = [table_one]
    output_path = Path("dataset_nam")
    output_path.mkdir(exist_ok=True)

    for artifact in artifacts:
        artifact(output_path)

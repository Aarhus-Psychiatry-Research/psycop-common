from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.cvd.cohort_examination.table_one.facade import table_one


class CVDArtifactFacade(Protocol):
    def __call__(self, output_dir: Path) -> None:
        ...


if __name__ == "__main__":
    run = MlflowClientWrapper().get_run(experiment_name="CVD", run_name="CVD layer 1, base")
    output_path = Path("dataset_name")
    output_path.mkdir(exist_ok=True)

    artifacts: Sequence[CVDArtifactFacade] = [
        lambda output_dir: table_one(run=run, output_dir=output_dir)
    ]
    for artifact in artifacts:
        artifact(output_path)

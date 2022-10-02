"""Sync best offline runs to wandb.

Automatically moves all wandb runs to an archive folder and deletes the
AUC logging file so a new hyperparameter search can be safely started.
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd

from psycopt2d.utils import AUC_LOGGING_FILE_PATH, PROJECT_ROOT


def get_best_run_ids(top_n: int) -> list[str]:
    """Get the run ids with the highest AUC.

    Args:
        top_n (int): Number of runs to return

    Returns:
        list[str]: list of run ids
    """
    df = pd.read_csv(AUC_LOGGING_FILE_PATH)
    return df.sort_values("auc", ascending=False)["run_id"].tolist()[:top_n]


def sync_single_run_to_wandb(run: Path, project: str) -> None:
    """Sync a single run to wandb.

    Args:
        run (Path): Run id
        project (str): Project name
    """
    subprocess.run(
        ["wandb", "sync", f"wandb/{str(run.name)}", "--project", f"{project}"],
        check=True,
    )


def sync_runs_to_wandb(runs: list[str], project: str) -> None:
    """Sync runs to wandb.

    Args:
        runs (list[str]): list of run ids
        project (str): Project name
    """

    def get_run_path(run: str) -> Path:
        return list((Path() / "wandb").glob(f"*{run}*"))[0]

    run_paths = [get_run_path(run) for run in runs]
    for run in run_paths:
        sync_single_run_to_wandb(run, project)


def archive_wandb_runs() -> None:
    """Move all wandb runs to an archive folder."""
    wandb_path = PROJECT_ROOT / "wandb"
    archive_path = wandb_path / "archive"
    archive_path.mkdir(exist_ok=True)
    for run in wandb_path.glob("*"):
        if run == archive_path:
            continue
        run.rename(archive_path / run.name)


def sync_best_runs_to_wandb(top_n: int, project: str) -> None:
    """Sync the best runs to wandb. This will move all wandb runs to an archive
    and delete the AUC logging file.

    Args:
        top_n (int): Number of runs to sync
        project (str): Project name
    """
    runs = get_best_run_ids(top_n)
    sync_runs_to_wandb(runs, project)
    archive_wandb_runs()
    AUC_LOGGING_FILE_PATH.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--project", type=str, default="psycopt2d")
    args = parser.parse_args()

    sync_best_runs_to_wandb(args.top_n, args.project)

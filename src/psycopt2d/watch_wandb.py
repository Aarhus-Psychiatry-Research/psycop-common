"""Watches the wandb directory for new files and uploads them to wandb."""
import argparse
import re
import subprocess
import time
from distutils.util import strtobool
from pathlib import Path
from typing import Union

import pandas as pd
import wandb
from wandb.apis.public import Api
from wandb.sdk.wandb_run import Run
from wasabi import msg

from psycopt2d.evaluation import evaluate_model
from psycopt2d.utils import MODEL_PREDICTIONS_PATH, PROJECT_ROOT, read_pickle

# Path to the wandb directory
WANDB_DIR = PROJECT_ROOT / "wandb"


class WandbWatcher:
    """Watch the wandb directory for new files and uploads them to wandb.
    Fully evaluates the best runs after a certain number of runs have been
    uploaded.

    Args:
        entity: The wandb entity to upload to (e.g. "psycop")
        project_name: The wandb project name to upload to (e.g. "psycopt2d")
        n_runs_before_eval: The number of runs to upload before evaluating the
            best runs.
        overtaci: Whether the script is running on overtaci. Determines where
            to look for the evaluation results.
    """

    def __init__(
        self,
        entity: str,
        project_name: str,
        n_runs_before_eval: int,
        overtaci: bool,
    ):
        self.entity = entity
        self.project_name = project_name
        self.overtaci = overtaci
        self.model_data_dir = (
            MODEL_PREDICTIONS_PATH / project_name
            if overtaci
            else PROJECT_ROOT / "evaluation_results"
        )

        self.n_runs_before_eval = n_runs_before_eval

        self.run_ids = []
        self.max_performance = 0

        self.archive_path = WANDB_DIR / "archive"
        self.archive_path.mkdir(exist_ok=True)

    def watch(self, timeout: int = 0) -> None:
        """Watch the wandb directory for new runs.

        Args:
            timeout: The timeout in minutes. If 0, the script will run
                indefinitely.
        """
        start_time = time.time()
        while start_time + timeout * 60 > time.time() or timeout == 0:
            self.get_new_runs_and_evaluate()
            time.sleep(20)

    def get_new_runs_and_evaluate(self) -> None:
        """Get new runs and evaluate the best runs."""
        self.upload_recent_runs()
        if len(self.run_ids) >= self.n_runs_before_eval:
            self.evaluate_best_runs()

    def upload_recent_runs(self) -> None:
        """Upload unarchived runs to wandb."""
        for run_folder in WANDB_DIR.glob("offline-run*"):
            run_id = self._get_run_id(run_folder)

            self._upload_run(run_folder)
            self._archive_run(run_folder)
            self.run_ids.append(run_id)

    def evaluate_best_runs(self) -> None:
        """Evaluate the best runs."""
        run_performances = {
            run_id: self._get_run_performance(run_id) for run_id in self.run_ids
        }

        for run_id, performance in run_performances.items():
            if performance > self.max_performance:
                msg.good(f"New record performance! AUC: {performance}")
                self.max_performance = performance
                self._do_evaluation(run_id)
        # reset run id tracker
        self.run_ids = []

    def archive_all_runs(self) -> None:
        """Archive all runs in the wandb directory."""
        for run_folder in WANDB_DIR.glob("offline-run*"):
            self._archive_run(run_folder)

    def _do_evaluation(self, run_id: str) -> None:
        """Do the full evaluation of the run and upload to wandb."""
        # get evaluation data
        eval_df, cfg, feature_importance_dict = self._get_eval_data(run_id)
        # infer required column names
        y_col_name = infer_outcome_col_name(df=eval_df, prefix="outc_")
        y_hat_prob_col_name = infer_y_hat_prop_col_name(df=eval_df)
        # get wandb run
        run = wandb.init(project=self.project_name, entity=self.entity, id=run_id)

        # run evaluation
        evaluate_model(
            cfg=cfg,
            eval_df=eval_df,
            y_col_name=y_col_name,
            y_hat_prob_col_name=y_hat_prob_col_name,
            run=run,
            feature_importance_dict=feature_importance_dict,
        )
        run.finish()

    def _get_eval_data(self, run_id: str) -> tuple:
        """Get the evaluation data for a single run."""
        run_eval_dir = self._get_run_evaluation_dir(run_id)

        eval_df = pd.read_parquet(run_eval_dir / "df.parquet")
        cfg = read_pickle(str(run_eval_dir / "cfg.pkl"))

        if (run_eval_dir / "feature_importance.pkl").exists():
            feature_importance = read_pickle(
                str(run_eval_dir / "feature_importance.pkl")
            )
        else:
            feature_importance = None
        return eval_df, cfg, feature_importance

    def _get_run_evaluation_dir(self, run_id: str) -> Path:
        """Get the evaluation path for a single run."""
        return list(self.model_data_dir.glob(f"*{run_id}*"))[0]

    def _upload_run(self, run: Path) -> None:
        """Upload a single run to wandb."""
        subprocess.run(
            ["wandb", "sync", str(run), "--project", self.project_name],
            check=True,
        )

    def _archive_run(self, run: Path) -> None:
        """Move a run to the archive folder."""
        run.rename(self.archive_path / run.name)

    def _get_run_id(self, run: Path) -> str:
        """Get the run id from the wandb directory."""
        return run.name.split("-")[-1]

    def _get_run_performance(self, run_id: str) -> float:
        """Get the performance of a single run and check if it failed."""
        run = self._get_wandb_run(run_id)
        if run.state == "failed":
            wandb.init(project=self.project_name, entity=self.entity, id=run_id)
            wandb.alert(title="Failed run", text=f"Run {run_id} failed.")
            return 0.0
        elif hasattr(run.summary, "roc_auc_unweighted"):
            return run.summary.roc_auc_unweighted
        else:
            msg.warn(f"Run {run_id} has no performance metric.")
            return 0.0

    def _get_wandb_run(self, run_id: str) -> Run:
        """Get the wandb run object from the run id."""
        return Api().run(f"{self.entity}/{self.project_name}/{run_id}")


def infer_outcome_col_name(df: pd.DataFrame, prefix: str = "outc_") -> str:
    """Infer the outcome column name from the dataframe."""
    outcome_name = [c for c in df.columns if c.startswith(prefix)]
    if len(outcome_name) == 1:
        return outcome_name[0]
    else:
        raise ValueError("More than one outcome inferred")


def infer_y_hat_prop_col_name(df: pd.DataFrame) -> str:
    """Infer the y_hat_prob column name from the dataframe."""
    y_hat_prob_name = [c for c in df.columns if c.startswith("y_hat_prob")]
    if len(y_hat_prob_name) == 1:
        return y_hat_prob_name[0]
    else:
        raise ValueError("More than one y_hat_prob inferred")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, help="Wandb entity", required=True)
    parser.add_argument(
        "--project_name", type=str, help="Wandb project name", required=True
    )
    parser.add_argument(
        "--n_runs_before_eval",
        type=int,
        help="Number of runs before first evaluation",
        required=True,
    )
    parser.add_argument(
        "--overtaci",
        type=lambda x: bool(strtobool(x)),
        help="Whether the script is run on Overtaci or not",
        required=True,
    )
    parser.add_argument("--timeout", type=int, help="Timeout in minutes", required=True)
    parser.add_argument(
        "--clean_wandb_dir",
        type=lambda x: bool(strtobool(x)),
        help="Archive all runs in the wandb dir before starting",
        required=True,
    )
    args = parser.parse_args()

    watcher = WandbWatcher(
        entity=args.entity,
        project_name=args.project_name,
        n_runs_before_eval=args.n_runs_before_eval,
        overtaci=args.overtaci,
    )
    if args.clean_wandb_dir:
        watcher.archive_all_runs()

    watcher.watch(timeout=args.timeout)

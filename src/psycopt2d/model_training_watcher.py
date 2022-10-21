"""Watches the wandb directory for new files and uploads them to wandb."""
import argparse
import subprocess
import time
from distutils.util import strtobool  # pylint: disable=deprecated-module
from pathlib import Path
from typing import Optional

import wandb
from wandb.apis.public import Api  # pylint: disable=no-name-in-module
from wandb.sdk.wandb_run import Run  # pylint: disable=no-name-in-module
from wasabi import msg

from psycopt2d.evaluation import evaluate_model
from psycopt2d.utils.utils import (
    MODEL_PREDICTIONS_PATH,
    PROJECT_ROOT,
    infer_outcome_col_name,
    infer_y_hat_prob_col_name,
    load_evaluation_data,
)

# Path to the wandb directory
WANDB_DIR = PROJECT_ROOT / "wandb"


class ModelTrainingWatcher:
    """Watch the wandb directory for new files and uploads them to wandb. Fully
    evaluates the best runs after a certain number of runs have been uploaded.

    Args:
        entity: The wandb entity to upload to (e.g. "psycop")
        project_name: The wandb project name to upload to (e.g. "psycopt2d")
        n_runs_before_eval: The number of runs to complete before evaluating the
            best runs.
        model_data_dir: Where to look for evaluation results.
        overtaci: Whether the script is running on overtaci. Determines where
            to look for the evaluation results.
    """

    def __init__(
        self,
        entity: str,
        project_name: str,
        n_runs_before_eval: int,
        model_data_dir: Path,
    ):
        self.entity = entity
        self.project_name = project_name
        self.model_data_dir = model_data_dir

        self.n_runs_before_eval = n_runs_before_eval

        # A queue for runs waiting to be uploaded to WandB
        self.run_id_eval_candidates_queue = []
        self.max_performance = 0

        self.archive_path = WANDB_DIR / "archive"
        self.archive_path.mkdir(exist_ok=True, parents=True)

    def watch(self, timeout_minutes: Optional[int] = None) -> None:
        """Watch the wandb directory for new runs.

        Args:
            timeout_minutes: The timeout in minutes. If None, the script will run
                indefinitely.
        """
        start_time = time.time()
        while (
            timeout_minutes is None or start_time + timeout_minutes * 60 > time.time()
        ):
            self.get_new_runs_and_evaluate()
            time.sleep(1)

    def get_new_runs_and_evaluate(self) -> None:
        """Get new runs and evaluate the best runs."""
        self.upload_unarchived_runs()
        if len(self.run_id_eval_candidates_queue) >= self.n_runs_before_eval:
            self.evaluate_best_runs()

    def _upload_run_dir(self, run_dir: Path) -> None:
        """Upload a single run to wandb."""
        subprocess.run(
            ["wandb", "sync", str(run_dir), "--project", self.project_name],
            check=True,
        )

    def _archive_run_dir(self, run_dir: Path) -> None:
        """Move a run to the archive folder."""
        run_dir.rename(target=self.archive_path / run_dir.name)

    def _get_run_id(self, run_dir: Path) -> str:
        """Get the run id from a run directory."""
        return run_dir.name.split("-")[-1]

    def upload_unarchived_runs(self) -> None:
        """Upload unarchived runs to wandb."""
        for run_folder in WANDB_DIR.glob(r"offline-run*"):
            # TODO: We need some kind of test here to figure out if the run is
            # still running or not. If it is still running, we should wait
            # until it is finished. Otherwise, we get a "permission denied" error.
            run_id = self._get_run_id(run_folder)

            self._upload_run_dir(run_folder)

            # TODO: If upload_run_dir fails, we should not archive the run.
            # use return from subprocess.run to check if it failed. See docs: https://docs.python.org/3/library/subprocess.html
            self._archive_run_dir(run_folder)
            self.run_id_eval_candidates_queue.append(run_id)

    def _get_run_evaluation_dir(self, run_id: str) -> Path:
        """Get the evaluation path for a single run."""
        return list(self.model_data_dir.glob(f"*{run_id}*"))[0]

    def _get_eval_data(self, run_id: str) -> tuple:
        """Get the evaluation data for a single run."""
        run_eval_dir = self._get_run_evaluation_dir(run_id)

        return load_evaluation_data(run_eval_dir)

    def _do_evaluation(self, run_id: str) -> None:
        """Do the full evaluation of the run and upload to wandb."""
        # get evaluation data
        eval_data = self._get_eval_data(run_id)
        # infer required column names
        y_col_name = infer_outcome_col_name(df=eval_data.df, prefix="outc_")
        y_hat_prob_col_name = infer_y_hat_prob_col_name(df=eval_data.df)
        # get wandb run
        run = wandb.init(project=self.project_name, entity=self.entity, id=run_id)

        # run evaluation
        evaluate_model(
            cfg=eval_data.cfg,
            eval_df=eval_data.df,
            y_col_name=y_col_name,
            y_hat_prob_col_name=y_hat_prob_col_name,
            run=run,
            feature_importance_dict=eval_data.feature_importance_dict,
        )
        run.finish()

    def _get_wandb_run(self, run_id: str) -> Run:
        """Get the wandb run object from the run id."""
        return Api().run(f"{self.entity}/{self.project_name}/{run_id}")

    def _get_run_performance(self, run_id: str) -> float:
        """Get the performance of a single run and check if it failed."""
        run = self._get_wandb_run(run_id)
        if "roc_auc_unweighted" in run.summary:
            return run.summary.roc_auc_unweighted
        msg.info(
            f"Run {run_id} has no performance metric. Pinging again at next eval time.",
        )
        return None

    def evaluate_best_runs(self) -> None:
        """Evaluate the best runs."""
        run_performances = {
            run_id: self._get_run_performance(run_id)
            for run_id in self.run_id_eval_candidates_queue
        }
        # sort runs by performance to not upload subpar runs
        run_performances = dict(
            sorted(run_performances.items(), key=lambda item: item[1], reverse=True),
        )
        # get runs with auc of None (attempted upload before run finished)
        unfinished_runs = [
            run_id for run_id, auc in run_performances.items() if auc is None
        ]

        for run_id, performance in run_performances.items():
            if performance > self.max_performance:
                msg.good(f"New record performance! AUC: {performance}")
                self.max_performance = performance
                self._do_evaluation(run_id)
        # reset run id queue and try to upload unfinished runs next time
        self.run_id_eval_candidates_queue = unfinished_runs

    def archive_all_runs(self) -> None:
        """Archive all runs in the wandb directory."""
        for run_folder in WANDB_DIR.glob("*run*"):
            self._archive_run_dir(run_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, help="Wandb entity", required=True)
    parser.add_argument(
        "--project_name",
        type=str,
        help="Wandb project name",
        required=True,
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

    def float_or_none(arg: str) -> Optional[float]:
        """Wrapper function to take float or none in argparse."""
        return None if arg == "None" else float(arg)

    parser.add_argument(
        "--timeout",
        type=float_or_none,
        help="""How long to run the watcher for. If None, keeps runnning until process
        is killed (e.g. receives SIGTERM())""",
    )
    parser.add_argument(
        "--clean_wandb_dir",
        type=lambda x: bool(strtobool(x)),
        help="Archive all runs in the wandb dir before starting",
        required=True,
    )
    args = parser.parse_args()

    model_data_dir = (
        MODEL_PREDICTIONS_PATH / args.project_name
        if args.overtaci
        else PROJECT_ROOT / "evaluation_results"
    )

    watcher = ModelTrainingWatcher(
        entity=args.entity,
        project_name=args.project_name,
        n_runs_before_eval=args.n_runs_before_eval,
        model_data_dir=model_data_dir,
    )
    if args.clean_wandb_dir:
        watcher.archive_all_runs()

    msg.info("Starting WandB watcher")
    watcher.watch(timeout_minutes=args.timeout)

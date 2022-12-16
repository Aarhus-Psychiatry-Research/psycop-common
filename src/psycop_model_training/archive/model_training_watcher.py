"""Watches the wandb directory for new files and uploads them to wandb."""
import argparse
import subprocess
import time
from collections import defaultdict
from distutils.util import strtobool  # pylint: disable=deprecated-module
from pathlib import Path
from typing import Any, Optional, Union

import wandb
from pydantic import BaseModel
from wandb.apis.public import Api  # pylint: disable=no-name-in-module
from wandb.sdk.wandb_run import Run  # pylint: disable=no-name-in-module
from wasabi import msg

from psycop_model_training.model_eval.dataclasses import ModelEvalData
from psycop_model_training.model_eval.evaluate_model import run_full_evaluation
from psycop_model_training.utils.config_schemas import FullConfigSchema
from psycop_model_training.utils.utils import (
    MODEL_PREDICTIONS_PATH,
    PROJECT_ROOT,
    load_evaluation_data,
)

WANDB_DIR = PROJECT_ROOT / "wandb"


def start_watcher(cfg: FullConfigSchema) -> subprocess.Popen:
    """Start a watcher."""
    return subprocess.Popen(  # pylint: disable=consider-using-with
        [
            "python",
            "src/psycop_model_training/model_training_watcher.py",
            "--entity",
            cfg.project.wandb.entity,
            "--project_name",
            cfg.project.name,
            "--n_runs_before_eval",
            str(cfg.project.watcher.n_runs_before_eval),
            "--overtaci",
            str(cfg.eval.save_model_predictions_on_overtaci),
            "--timeout",
            "None",
            "--clean_wandb_dir",
            str(cfg.project.watcher.archive_all),
            "--verbose",
            "True",
        ],
    )


class RunInformation(BaseModel):
    """Information about a wandb run."""

    # Attributes must be optional since runs can be uploaded,
    # without having been sufficiently validated.
    run_id: str
    auc: Optional[float]
    lookbehind_days: Optional[Union[int, list[int]]]
    lookahead_days: Optional[int]

    # Concatenated lookbehind_days and lookahead_days string for
    # max_performances scoreboard. Allows to only fully evaluate
    # the models that set new high scores.
    lookahead_lookbehind_combined: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if (
            self.lookahead_lookbehind_combined is None
            and self.lookbehind_days is not None
        ):
            self.lookahead_lookbehind_combined = f"lookahead:{str(self.lookahead_days)}_lookbehind:{str(self.lookbehind_days)}"


class ModelTrainingWatcher:  # pylint: disable=too-many-instance-attributes
    """Watch the wandb directory for new files and uploads them to wandb. Fully
    evaluates the best runs after a certain number of runs have been uploaded.

    Args:
        entity: The wandb entity to upload to (e.g. "psycop")
        project_name: The wandb project name to upload to (e.g. "psycop_model_training")
        n_runs_before_eval: The number of runs to complete before evaluating the
            best runs.
        model_data_dir: Where to look for evaluation results.
        overtaci: Whether the script is running on overtaci. Determines where
            to look for the evaluation results.
        verbose: Whether to print verbose output.
    """

    def __init__(
        self,
        entity: str,
        project_name: str,
        n_runs_before_eval: int,
        model_data_dir: Path,
        verbose: bool = False,
    ):
        self.entity = entity
        self.project_name = project_name
        self.model_data_dir = model_data_dir

        self.n_runs_before_eval = n_runs_before_eval

        self.verbose = verbose
        # A queue for runs waiting to be uploaded to WandB
        self.run_id_eval_candidates_queue: list[str] = []
        # max performance by lookbehind/-ahead combination
        self.max_performances: dict[str, float] = defaultdict(lambda: 0.0)

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

    def _archive_run_dir(self, run_dir: Path) -> None:
        """Move a run to the archive folder."""
        run_dir.rename(target=self.archive_path / run_dir.name)

    def _get_run_id(self, run_dir: Path) -> str:
        """Get the run id from a run directory."""
        return run_dir.name.split("-")[-1]

    def _upload_run_dir(self, run_dir: Path) -> str:
        """Upload a single run to wandb."""
        # get stdout from subprocess.run
        proc = subprocess.run(
            ["wandb", "sync", str(run_dir), "--project", self.project_name],
            check=True,
            capture_output=True,
        )

        stdout = proc.stdout.decode("utf-8")

        if self.verbose:
            msg.info(f"Watcher: {stdout}")
        return stdout

    def _get_run_evaluation_data_dir(self, run_id: str) -> Path:
        """Get the evaluation path for a single run."""
        return list(self.model_data_dir.glob(f"*{run_id}*"))[0]

    def _get_eval_data(self, run_id: str) -> ModelEvalData:
        """Get the evaluation data for a single run."""
        run_eval_dir = self._get_run_evaluation_data_dir(run_id)

        return load_evaluation_data(run_eval_dir)

    def _do_evaluation(self, run_id: str) -> None:
        """Do the full evaluation of the run and upload to wandb."""
        # get evaluation data
        eval_data = self._get_eval_data(run_id=run_id)

        run: Run = wandb.init(project=self.project_name, entity=self.entity, id=run_id)  # type: ignore

        # run evaluation
        run_full_evaluation(
            cfg=eval_data.cfg,
            eval_dataset=eval_data.eval_dataset,
            pipe_metadata=eval_data.pipe_metadata,
            run=run,
            save_dir=PROJECT_ROOT / "wandb" / run.name / ".tmp",
        )

        run.finish()

    def _get_wandb_run(self, run_id: str) -> Run:
        """Get the wandb run object from the run id."""
        return Api().run(f"{self.entity}/{self.project_name}/{run_id}")

    def _get_run_wandb_dir(self, run_id: str) -> Path:
        return list(WANDB_DIR.glob(f"*offline-run*{run_id}*"))[0]

    def _get_run_attribute(self, run: Run, attribute: str) -> Any:
        """Get an attribute from a wandb run."""
        if attribute in run.summary:  # type: ignore
            return run.summary[attribute]
        if self.verbose:
            msg.info(
                f"Run {run.id} has no attribute {attribute}. Pinging again at next eval time.",
            )
        return None

    def _evaluate_and_archive_finished_runs(
        self,
        run_information: list[RunInformation],
    ) -> None:
        """Evaluate the finished runs.

        Test their performance against the current maximum for each
        lookbehind/-ahead days, and fully evaluate the best performing.
        Move all wandb run dirs to the archive folder.
        """
        finished_runs: list[RunInformation] = [
            run_info
            for run_info in run_information
            if run_info.auc and run_info.lookahead_lookbehind_combined
        ]
        # sort to only upload the best in in each group
        finished_runs.sort(
            key=lambda run_info: (
                run_info.lookahead_lookbehind_combined,
                run_info.auc,
            ),
            reverse=True,
        )

        if finished_runs:
            for run_info in finished_runs:
                if (
                    run_info.auc  # type: ignore
                    > self.max_performances[
                        run_info.lookahead_lookbehind_combined  # type: ignore
                    ]
                ):
                    msg.good(
                        f"New record performance for {run_info.lookahead_lookbehind_combined}! AUC: {run_info.auc}",
                    )
                    self.max_performances[
                        run_info.lookahead_lookbehind_combined  # type: ignore
                    ] = run_info.auc  # type: ignore
                    self._do_evaluation(run_info.run_id)
                self._archive_run_dir(run_dir=self._get_run_wandb_dir(run_info.run_id))

    def _get_unfinished_run_ids(
        self,
        run_information: list[RunInformation],
    ) -> list[str]:
        """Get the run ids of the unfinished runs."""
        return [run_info.run_id for run_info in run_information if run_info.auc is None]

    def _get_run_information(self, run_id: str) -> RunInformation:
        """Get the run information for a single run."""
        run = self._get_wandb_run(run_id)
        return RunInformation(
            run_id=run_id,
            auc=self._get_run_attribute(run, "roc_auc_unweighted"),
            lookbehind_days=self._get_run_attribute(run, "lookbehind"),
            lookahead_days=self._get_run_attribute(run, "lookahead"),
        )

    def _get_run_information_for_all_in_queue(self):
        """Get the performance and information of all runs in the evaluation
        queue."""
        return [
            self._get_run_information(run_id)
            for run_id in self.run_id_eval_candidates_queue
        ]

    def get_new_runs_and_evaluate(self) -> None:
        """Get new runs and evaluate the best runs."""
        self.upload_unarchived_runs()

        if len(self.run_id_eval_candidates_queue) >= self.n_runs_before_eval:
            run_infos = self._get_run_information_for_all_in_queue()
            self.run_id_eval_candidates_queue = self._get_unfinished_run_ids(
                run_information=run_infos,
            )
            self._evaluate_and_archive_finished_runs(run_information=run_infos)

    def upload_unarchived_runs(self) -> None:
        """Upload unarchived runs to wandb. Only adds runs that have finished
        training to the evaluation queue.

        Raises:
            ValueError: If wandb sync failed
        """
        for run_folder in WANDB_DIR.glob(r"offline-run*"):
            run_id = self._get_run_id(run_folder)

            wandb_sync_stdout = self._upload_run_dir(run_folder)

            if "... done" not in wandb_sync_stdout:
                if ".wandb file is empty" not in wandb_sync_stdout:
                    raise ValueError(
                        f"wandb sync failed, returned: {wandb_sync_stdout}",
                    )
                if self.verbose:
                    msg.warn(f"Run {run_id} is still running. Skipping.")
                continue

            self.run_id_eval_candidates_queue.append(run_id)

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
    parser.add_argument(
        "--verbose",
        type=lambda x: bool(strtobool(x)),
        help="Whether to print verbose messages (default: False)",
        required=False,
        default=False,
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
        verbose=args.verbose,
    )

    if args.clean_wandb_dir:
        watcher.archive_all_runs()

    msg.info("Watcher: Starting WandB watcher")
    watcher.watch(timeout_minutes=args.timeout)

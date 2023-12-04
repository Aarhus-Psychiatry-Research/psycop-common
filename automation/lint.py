from pathlib import Path

from invoke import Context, task

from .error_handling import exit_if_error_in_stdout
from .git import _add_commit, is_uncommitted_changes
from .logger import echo_header, msg_type
from .windows import NOT_WINDOWS


def pre_commit(c: Context, auto_fix: bool):
    """Run pre-commit checks."""

    # Essential to have a clean working directory before pre-commit to avoid committing
    # heterogenous files under a "style: linting" commit
    if auto_fix and is_uncommitted_changes(c):
        print(
            f"{msg_type.WARN} Your git working directory is not clean. Stash or commit before running pre-commit.",
        )
        exit(1)

    echo_header(f"{msg_type.CLEAN} Running pre-commit checks")
    pre_commit_cmd = "pre-commit run --all-files"
    result = c.run(pre_commit_cmd, pty=NOT_WINDOWS, warn=True)

    if ("fixed" in result.stdout or "reformatted" in result.stdout) and auto_fix:
        for _ in range(2):
            # Run 3 times to ensure ruff/black interaction is resolved
            # E.g. ruff adding a trailing comma can make black reformat
            # the file again
            print(f"{msg_type.DOING} Fixed errors, re-running pre-commit checks\n\n")
            result = c.run(pre_commit_cmd, pty=NOT_WINDOWS, warn=True)

            if not ("fixed" in result.stdout or "reformatted" in result.stdout):
                break

        exit_if_error_in_stdout(result)
        _add_commit(c, msg="style: auto-fixes from pre-commit")
    else:
        if result.return_code != 0:
            print(f"{msg_type.FAIL} Pre-commit checks failed")
            exit(1)


def test_for_venv(c: Context):
    """Test if the user is in a virtual environment."""
    # Check if in docker environment by checking if the /.dockerenv file exists
    IN_DOCKER = Path("/.dockerenv").exists()
    if IN_DOCKER:
        print("Running in docker, not checking for virtual environment.")
        return

    if NOT_WINDOWS:
        python_path = c.run("which python", pty=NOT_WINDOWS, hide=True).stdout

        if python_path is None or "venv" not in python_path:
            print(f"\n{msg_type.FAIL} Not in a virtual environment.\n")
            print("Activate your virtual environment and try again. \n")
            exit(1)
    else:
        print("Running on Windows, not checking for virtual environment.")

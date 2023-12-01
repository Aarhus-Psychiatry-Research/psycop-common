import multiprocessing
import platform
import re
import shutil
from pathlib import Path
from typing import Optional

from invoke import Context, Result, task

from automation.git import is_uncommitted_changes
from automation.windows import NOT_WINDOWS


def exit_if_error_in_stdout(result: Result):
    # Find N remaining using regex

    if "error" in result.stdout:
        errors_remaining = re.findall(r"\d+(?=( remaining))", result.stdout)[
            0
        ]  # testing
        if errors_remaining != "0":
            exit(0)

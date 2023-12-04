from pathlib import Path

from invoke import Context, Result, task

from tasks import create_pr

from .environment import NOT_WINDOWS, on_ovartaci
from .logger import echo_header, msg_type


def update_pr(c: Context):
    if not on_ovartaci():
        echo_header(f"{msg_type.COMMUNICATE} Syncing PR")
        # Get current branch name
        branch_name = Path(".git/HEAD").read_text().split("/")[-1].strip()
        pr_result: Result = c.run(
            "gh pr list --state OPEN",
            pty=False,
            hide=True,
        )

        if branch_name not in pr_result.stdout:
            create_pr(c)

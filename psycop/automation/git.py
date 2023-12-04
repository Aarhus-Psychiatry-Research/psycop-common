from pathlib import Path
from typing import Optional

from invoke import Context, Result

from .environment import NOT_WINDOWS
from .logger import echo_header, msg_type


def is_uncommitted_changes(c: Context) -> bool:
    git_status_result: Result = c.run(
        "git status --porcelain",
        pty=NOT_WINDOWS,
        hide=True,
    )

    uncommitted_changes = git_status_result.stdout != ""
    return uncommitted_changes


def filetype_modified_since_head(c: Context, file_suffix: str) -> bool:
    files_modified_since_main = c.run(
        "git diff --name-only origin/main",
        hide=True,
    ).stdout.splitlines()

    if any(file.endswith(file_suffix) for file in files_modified_since_main):
        return True

    return False


def add_commit(c: Context, msg: Optional[str] = None):
    print(f"{msg_type.DOING} Adding and committing changes")
    c.run("git add .")

    if msg is None:
        msg = input("Commit message [--a to amend previous commit]: ")

    if "--a" in msg:
        c.run("git commit --amend --reuse-message=HEAD", pty=NOT_WINDOWS, hide=True)
        print(f"{msg_type.GOOD} Commit amended")
    elif msg == "-a":
        print(
            f"{msg_type.FAIL} You typed '-a'. Did you mean '--a' to amend the previous commit?",
        )
    else:
        c.run(f'git commit -m "{msg}"', pty=NOT_WINDOWS, hide=True)
        print(f"{msg_type.GOOD} Changes added and committed")


def add_and_commit(c: Context, msg: Optional[str] = None):
    """Add and commit all changes."""
    if is_uncommitted_changes(c):
        uncommitted_changes_descr = c.run(
            "git status --porcelain",
            pty=NOT_WINDOWS,
            hide=True,
        ).stdout

        echo_header(
            f"{msg_type.WARN} Uncommitted changes detected",
        )

        for line in uncommitted_changes_descr.splitlines():
            print(f"    {line.strip()}")
        print("\n")
        add_commit(c, msg=msg)


def branch_exists_on_remote(c: Context) -> bool:
    branch_name = Path(".git/HEAD").read_text().split("/")[-1].strip()

    branch_exists_result: Result = c.run(
        f"git ls-remote --heads origin {branch_name}",
        hide=True,
    )

    return branch_name in branch_exists_result.stdout


def push_to_branch(c: Context):
    echo_header(f"{msg_type.SYNC} Syncing branch with remote")

    if not branch_exists_on_remote(c):
        c.run("git push --set-upstream origin HEAD")
    else:
        print("Pushing")
        c.run("git push")

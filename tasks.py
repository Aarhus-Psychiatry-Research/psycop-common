"""
This project uses Invoke (pyinvoke.org) for task management.
Install it via:

```
pip install invoke
```

And then run:

```
inv --list
```

If you do not wish to use invoke you can simply delete this file.
"""


import multiprocessing
import platform
import re
import shutil
from pathlib import Path
from typing import Optional

from invoke import Context, Result, task

# Extract supported python versions from the pyproject.toml classifiers key
SUPPORTED_PYTHON_VERSIONS = [
    line.split("::")[-1].strip().replace('"', "").replace(",", "")
    for line in Path("pyproject.toml").read_text().splitlines()
    if "Programming Language :: Python ::" in line
]

NOT_WINDOWS = platform.system() != "Windows"


def filetype_modified_since_head(c: Context, file_suffix: str) -> bool:
    files_modified_since_main = c.run(
        "git diff --name-only origin/main",
        hide=True,
    ).stdout.splitlines()

    if any(file.endswith(file_suffix) for file in files_modified_since_main):
        return True

    return False


def on_ovartaci() -> bool:
    import platform

    if platform.node() == "RMAPPS1279":
        print(f"\n{msg_type.GOOD} On Ovartaci")
        return True

    print(f"\n{msg_type.GOOD} Not on Ovartaci")
    return False


def echo_header(msg: str):
    print(f"\n--- {msg} ---")


class MsgType:
    # Emojis have to be encoded as bytes to not break the terminal on Windows
    @property
    def DOING(self) -> str:
        return b"\xf0\x9f\xa4\x96".decode() if NOT_WINDOWS else "DOING:"

    @property
    def GOOD(self) -> str:
        return b"\xe2\x9c\x85".decode() if NOT_WINDOWS else "DONE:"

    @property
    def FAIL(self) -> str:
        return b"\xf0\x9f\x9a\xa8".decode() if NOT_WINDOWS else "FAILED:"

    @property
    def WARN(self) -> str:
        return b"\xf0\x9f\x9a\xa7".decode() if NOT_WINDOWS else "WARNING:"

    @property
    def SYNC(self) -> str:
        return b"\xf0\x9f\x9a\x82".decode() if NOT_WINDOWS else "SYNCING:"

    @property
    def PY(self) -> str:
        return b"\xf0\x9f\x90\x8d".decode() if NOT_WINDOWS else ""

    @property
    def CLEAN(self) -> str:
        return b"\xf0\x9f\xa7\xb9".decode() if NOT_WINDOWS else "CLEANING:"

    @property
    def TEST(self) -> str:
        return b"\xf0\x9f\xa7\xaa".decode() if NOT_WINDOWS else "TESTING:"

    @property
    def COMMUNICATE(self) -> str:
        return b"\xf0\x9f\x93\xa3".decode() if NOT_WINDOWS else "COMMUNICATING:"

    @property
    def EXAMINE(self) -> str:
        return b"\xf0\x9f\x94\x8d".decode() if NOT_WINDOWS else "VIEWING:"


msg_type = MsgType()


def git_init(c: Context, branch: str = "main"):
    """Initialize a git repository if it does not exist yet."""
    # If no .git directory exits
    if not Path(".git").exists():
        echo_header(f"{msg_type.DOING} Initializing Git repository")
        c.run(f"git init -b {branch}")
        c.run("git add .")
        c.run("git commit -m 'Init'")
        print(f"{msg_type.GOOD} Git repository initialized")
    else:
        print(f"{msg_type.GOOD} Git repository already initialized")


def setup_venv(
    c: Context,
    python_path: str,
    venv_name: Optional[str] = None,
) -> str:
    """Create a virtual environment if it does not exist yet.

    Args:
        c: The invoke context.
        python_path: The python executable to use.
        venv_name: The name of the virtual environment. Defaults to ".venv".
    """
    if venv_name is None:
        venv_name = ".venv"

    if not Path(venv_name).exists():
        echo_header(
            f"{msg_type.DOING} Creating virtual environment using {msg_type.PY}:{python_path}",
        )
        c.run(f"{python_path} -m venv {venv_name}")
        print(f"{msg_type.GOOD} Virtual environment created")
    else:
        print(f"{msg_type.GOOD} Virtual environment already exists")
    return venv_name


def _add_commit(c: Context, msg: Optional[str] = None):
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


def is_uncommitted_changes(c: Context) -> bool:
    git_status_result: Result = c.run(
        "git status --porcelain",
        pty=NOT_WINDOWS,
        hide=True,
    )

    uncommitted_changes = git_status_result.stdout != ""
    return uncommitted_changes


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
        _add_commit(c, msg=msg)


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


@task
def create_pr(c: Context):
    pr_result: Result = c.run(
        "gh pr view --json url -q '.url'",
        pty=False,
        hide=True,
    )
    if "no pull requests found" in pr_result.stdout:
        pr_title = c.run(
            "$$(git rev-parse --abbrev-ref HEAD | tr -d '[:digit:]' | tr '-' ' ')",
        ).stdout.strip()

        c.run(
            f"gh pr create --title {pr_title} --body 'Automatically created PR from invoke' -w",
            pty=NOT_WINDOWS,
        )
        print(f"{msg_type.GOOD} PR created")
    else:
        print(f"{msg_type.GOOD} PR already exists at: {pr_result.stdout}")


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


def exit_if_error_in_stdout(result: Result):
    # Find N remaining using regex

    if "error" in result.stdout:
        errors_remaining = re.findall(r"\d+(?=( remaining))", result.stdout)[
            0
        ]  # testing
        if errors_remaining != "0":
            exit(0)


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


@task
def qtypes(c: Context):
    """Run static type checks."""
    if filetype_modified_since_head(c, ".py"):
        static_type_checks(c)
    else:
        print("🟢 No python files modified since main, skipping static type checks")


@task
def static_type_checks(c: Context):
    if not on_ovartaci():
        echo_header(f"{msg_type.CLEAN} Running static type checks")
        c.run("pyright psycop/", pty=NOT_WINDOWS)
    else:
        print(
            f"{msg_type.FAIL}: Cannot install pyright on Ovartaci, skipping static type checks",
        )


def get_python_path(preferred_version: str) -> Optional[str]:
    """Get path to python executable."""
    preferred_version_path = shutil.which(f"python{preferred_version}")

    if preferred_version_path is not None:
        return preferred_version_path

    print(
        f"{msg_type.WARN}: python{preferred_version} not found, continuing with default python version",
    )
    return shutil.which("python")


@task
def setup(c: Context, python_path: Optional[str] = None):
    """Confirm that a git repo exists and setup a virtual environment.

    Args:
        c: Invoke context
        python_path: Path to the python executable to use for the virtual environment. Uses the return value of `which python` if not provided.
    """
    git_init(c)

    if python_path is None:
        # get path to python executable
        python_path = get_python_path(preferred_version="3.10")
        if not python_path:
            print(f"{msg_type.FAIL} Python executable not found")
            exit(1)
    venv_name = setup_venv(c, python_path=python_path)

    print(
        f"{msg_type.DOING} Activate your virtual environment by running: \n\n\t\t source {venv_name}/bin/activate \n",
    )


@task(iterable="pytest_args")
def test(
    c: Context,
    pytest_args: list[str] = [],  # noqa
):
    """Run tests"""
    # Invoke requires list as type hints, but does not support lists as default arguments.
    # Hence this super weird type hint and default argument for the python_versions arg.
    echo_header(f"{msg_type.TEST} Running tests")

    n_cores = multiprocessing.cpu_count()

    if not pytest_args:
        pytest_args = [
            "psycop",
            f"-n {min([n_cores-2, 8])}",
            "-rfE",
            "--failed-first",
            "-p no:cov",
            "--disable-warnings",
            "-q",
            "--durations=5",
        ]

    pytest_arg_str = " ".join(pytest_args)

    command = f"pytest {pytest_arg_str}"
    test_result: Result = c.run(
        command,
        warn=True,
        pty=NOT_WINDOWS,
    )

    # If "failed" in the pytest results
    failed_tests = [line for line in test_result.stdout if line.startswith("FAILED")]

    if len(failed_tests) > 0:
        print("\n\n\n")
        echo_header("Failed tests")
        print("\n\n\n")
        echo_header("Failed tests")

        for line in failed_tests:
            # Remove from start of line until /test_
            line_sans_prefix = line[line.find("test_") :]

            # Keep only that after ::
            line_sans_suffix = line_sans_prefix[line_sans_prefix.find("::") + 2 :]
            print(f"FAILED {msg_type.FAIL} #{line_sans_suffix}     ")

    if test_result.return_code != 0:
        exit(test_result.return_code)


@task
def create_pr_from_staged_changes(c: Context):
    start_branch_name = c.run(
        "git rev-parse --abbrev-ref HEAD",
        hide=True,
    ).stdout.strip()

    pr_title = input("Enter PR title: ")
    branch_title = re.sub(r"[\-\s\:\,]", "_", pr_title).lower()
    c.run("git checkout main")
    c.run(f"git checkout -b {branch_title}")
    c.run(f"git commit -m '{pr_title}'")
    push_to_branch(c)
    update_pr(c)

    checkout_previous_branch = input("Checkout previous branch? [y/n]: ")
    if checkout_previous_branch.lower() == "y":
        c.run(f"git checkout {start_branch_name}")


@task
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


@task
def test_for_rej(c: Context):
    # Get all paths in current directory or subdirectories that end in .rej
    c = c
    search_dirs = [
        d
        for d in Path().iterdir()
        if d.is_dir()
        and not (
            "venv" in d.name
            or ".git" in d.name
            or "build" in d.name
            or ".tox" in d.name
            or "cache" in d.name
            or "vscode" in d.name
            or "wandb" in d.name
        )
    ]
    print(
        f"Looking for .rej files in the current dir and {[d.name for d in search_dirs]}",
    )

    # Get top_level rej files
    rej_files = list(Path().glob("*.rej"))

    for d in search_dirs:
        rej_files_in_dir = list(d.rglob("*.rej"))
        rej_files += rej_files_in_dir

    if len(rej_files) > 0:
        print(f"\n{msg_type.FAIL} Found .rej files leftover from cruft update.\n")
        for file in rej_files:
            print(f"    /{file}")
        print("\nResolve the conflicts and try again. \n")
        exit(1)
    else:
        print(f"{msg_type.GOOD} No .rej files found.")


@task
def lint(c: Context, auto_fix: bool = False):
    """Lint the project."""
    test_for_venv(c)
    test_for_rej(c)
    pre_commit(c=c, auto_fix=auto_fix)
    print("✅✅✅ Succesful linting! ✅✅✅")


@task
def pr(c: Context, auto_fix: bool = True, create_pr: bool = True):
    """Run all checks and update the PR."""
    add_and_commit(c)
    if create_pr:
        try:
            update_pr(c)
        except Exception as e:
            print(f"{msg_type.FAIL} Could not update PR: {e}. Continuing.")

    lint(c, auto_fix=auto_fix)
    push_to_branch(c)
    static_type_checks(c)
    test(c)


@task
def qtest(c: Context):
    if any(filetype_modified_since_head(c, suffix) for suffix in (".py", ".cfg")):
        test(
            c,
            pytest_args=[
                "psycop",
                "-rfE",
                "--failed-first",
                "-p no:cov",
                "-p no:xdist",
                "--disable-warnings",
                "-q",
                "--durations=5",
                "--testmon",
            ],
        )
        print("✅✅✅ Tests ran succesfully! ✅✅✅")
    else:
        print("🟢 No python files modified since main, skipping tests")


# TODO: #390 Make more durable testmon implementation


@task
def qpr(c: Context, auto_fix: bool = True, create_pr: bool = True):
    """Run all checks and update the PR."""
    add_and_commit(c)
    if create_pr:
        try:
            update_pr(c)
        except Exception as e:
            print(f"{msg_type.FAIL} Could not update PR: {e}. Continuing.")

    lint(c, auto_fix=auto_fix)
    push_to_branch(c)
    qtest(c)
    qtypes(c)


@task
def docs(c: Context, view: bool = False, view_only: bool = False):
    """
    Build and view docs. If neither build or view are specified, both are run.
    """
    if not view_only:
        echo_header(f"{msg_type.DOING}: Building docs")
        c.run("tox -e docs")

    if view or view_only:
        echo_header(f"{msg_type.EXAMINE}: Opening docs in browser")
        # check the OS and open the docs in the browser
        if platform.system() == "Windows":
            c.run("start docs/_build/html/index.html")
        else:
            c.run("open docs/_build/html/index.html")


@task
def update_deps(c: Context):
    c.run("pip install --upgrade -r requirements.txt")

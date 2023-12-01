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

from automation.git import is_uncommitted_changes
from automation.windows import NOT_WINDOWS


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


@task(aliases=("static_type_checks",))
def types(c: Context):
    if not on_ovartaci():
        echo_header(f"{msg_type.CLEAN} Running static type checks")
        c.run("pyright psycop/", pty=NOT_WINDOWS)
    else:
        print(
            f"{msg_type.FAIL}: Cannot install pyright on Ovartaci, skipping static type checks",
        )


@task
def qtypes(c: Context):
    """Run static type checks."""
    if filetype_modified_since_head(c, ".py"):
        types(c)
    else:
        print("🟢 No python files modified since main, skipping static type checks")


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
def qtest(c: Context):
    """Quick tests, runs a subset of the tests using testmon"""
    # TODO: #390 Make more durable testmon implementation
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


@task(aliases=("format", "fmt"))
def lint(c: Context, auto_fix: bool = False):
    """Lint the project."""
    test_for_venv(c)
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
    types(c)
    test(c)


@task
def qpr(c: Context, auto_fix: bool = True, create_pr: bool = True):
    """Run all checks and update the PR, using heuristics for more speed."""
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
def vulnerability_scan(c: Context):
    requirements_files = Path().parent.glob("*requirements.txt")
    for requirements_file in requirements_files:
        c.run(
            f"snyk test --file={requirements_file} --package-manager=pip",
            pty=NOT_WINDOWS,
        )

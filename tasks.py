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


import multiprocessing  # noqa: I001
from pathlib import Path

from invoke import Context, Result, task

from psycop.automation.environment import NOT_WINDOWS, on_ovartaci
from psycop.automation.git import (
    add_and_commit,
    filetype_modified_since_main,
    push_to_branch,
)
from psycop.automation.lint import pre_commit
from psycop.automation.logger import echo_header, msg_type


@task
def install_requirements(c: Context):
    requirements_files = Path().parent.glob("*requirements.txt")
    requirements_string = " -r ".join([str(file) for file in requirements_files])
    c.run(f"pip install -r {requirements_string}")


@task(aliases=("static_type_checks", "type_check"))
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
    if filetype_modified_since_main(c, r"\.py$"):
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
    if any(filetype_modified_since_main(c, suffix) for suffix in (r"\.py$", r"\.cfg$")):
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
    pre_commit(c=c, auto_fix=auto_fix)
    print("✅✅✅ Succesful linting! ✅✅✅")


@task(aliases=("mm",))
def merge_main(c: Context):
    print(f"{msg_type.DOING} Merging main into current branch")
    c.run("git fetch")
    c.run("git merge --no-edit origin/main")
    print("✅✅✅ Merged main into current branch ✅✅✅")


@task(aliases=("am",))
def automerge(c: Context):
    c.run("gh pr merge --merge --auto --delete-branch")


@task(aliases=("vuln",))
def vulnerability_scan(c: Context, modified_files_only: bool = False):
    requirements_files = Path().parent.glob("*requirements.txt")

    if modified_files_only and not filetype_modified_since_main(
        c,
        r"requirements\.txt$",
    ):
        print(
            "🟢 No requirements.txt files modified since main, skipping vulnerability scan",
        )
        return

    for requirements_file in requirements_files:
        c.run(
            f"snyk test --file={requirements_file} --package-manager=pip",
            pty=NOT_WINDOWS,
        )


@task
def create_pr(c: Context):
    """
    Created a PR, does not run tests or similar
    """
    try:
        pr_result: Result = c.run(
            "gh pr view --json url -q '.url'",
            pty=False,
            hide=True,
        )
        print(f"{msg_type.GOOD} PR already exists at: {pr_result.stdout}")
    except Exception:
        branch_title = c.run(
            "git rev-parse --abbrev-ref HEAD",
            hide=True,
        ).stdout.strip()
        preprocessed_pr_title = branch_title.split("-")[1:]
        preprocessed_pr_title[0] = f"{preprocessed_pr_title[0]}:"
        pr_title = " ".join(preprocessed_pr_title)

        c.run(
            f'gh pr create --title "{pr_title}" --body "Automatically created PR from invoke" -w',
            pty=NOT_WINDOWS,
        )
        print(f"{msg_type.GOOD} PR created")


@task(aliases=("pr",))
def check_and_submit_pull_request(
    c: Context,
    auto_fix: bool = True,
):
    """Run all checks and update the PR."""
    add_and_commit(c)
    try:
        create_pr(c)
    except Exception as e:
        print(f"{msg_type.FAIL} Could not update PR: {e}. Continuing.")

    lint(c, auto_fix=auto_fix)
    push_to_branch(c)
    types(c)
    test(c)


@task(aliases=("qpr",))
def quick_check_and_submit_pull_request(
    c: Context,
    auto_fix: bool = True,
):
    """Run all checks and update the PR, using heuristics for more speed."""
    add_and_commit(c)
    try:
        create_pr(c)
    except Exception as e:
        print(f"{msg_type.FAIL} Could not update PR: {e}. Continuing.")

    lint(c, auto_fix=auto_fix)
    push_to_branch(c)
    qtest(c)
    qtypes(c)

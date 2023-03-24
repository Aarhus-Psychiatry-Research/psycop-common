from pathlib import Path
from typing import Optional

from invoke import Context, Result, task


def echo_header(msg: str):
    print(f"\n--- {msg} ---")


@task
def setup(c: Context, python_version: str = "3.9"):
    git_init(c)
    setup_venv(c, python_version=python_version)
    install(c)


def git_init(c: Context):
    # If no .git directory exits
    if not Path(".git").exists():
        echo_header("🔨 Initializing Git repository")
        c.run("git init")
        c.run("git add .")
        c.run("git commit -m 'Initial commit'")
        print("✅ Git repository initialized")
    else:
        print("✅ Git repository already initialized")


def setup_venv(
    c: Context,
    python_version: str,
):
    venv_name = f'.venv{python_version.replace(".", "")}'

    if not Path(venv_name).exists():
        echo_header("🔨 Creating virtual environment")
        c.run(f"python{python_version} -m venv {venv_name}")
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")

    c.run(f"source {venv_name}/bin/activate")


@task
def install(c: Context):
    echo_header("🔨 Installing project")
    c.run("pip install -e '.[dev,tests]'")


@task
def update(c: Context):
    echo_header("🔨 Updating project")
    c.run("pip install --upgrade -e '.[dev,tests]'")


@task
def test(c: Context):
    echo_header("🧪 Running tests")
    test_result: Result = c.run(
        "pytest -n auto -rfE --failed-first -p no:typeguard -p no:cov --disable-warnings -q",
        warn=True,
        pty=True,
    )

    # If "failed" in the pytest results
    if "failed" in test_result.stdout:
        print("\n\n\n")
        echo_header("Failed tests")

        # Get lines with "FAILED" in them from the .pytest_results file
        failed_tests = [
            line
            for line in Path("tests/.pytest_results").read_text().splitlines()
            if line.startswith("FAILED")
        ]

        for line in failed_tests:
            # Remove from start of line until /test_
            line_sans_prefix = line[line.find("test_") :]

            # Keep only that after ::
            line_sans_suffix = line_sans_prefix[line_sans_prefix.find("::") + 2 :]
            print(f"FAILED 🚨 #{line_sans_suffix}     ")

        exit(0)


def add_commit(c: Context, msg: Optional[str] = None):
    print("🔨 Adding and committing changes")
    c.run("git add .")

    if msg is None:
        msg = input("Commit message: ")
    c.run(f'git commit -m "{msg}"')


def add_and_commit(c: Context):
    git_status_result: Result = c.run(
        "git status --porcelain",
        pty=True,
        hide=True,
    )

    uncommitted_changes = git_status_result.stdout != ""
    uncommitted_changes_descr = git_status_result.stdout

    if uncommitted_changes:
        echo_header(
            "🚧 Uncommitted changes detected",
        )

        input("Press enter to add and commit the changes...")

        for line in uncommitted_changes_descr.splitlines():
            print(f"    {line.strip()}")
        print("\n")
        add_commit(c)


@task
def pr(c: Context):
    add_and_commit(c)
    lint(c)
    test(c)
    sync_with_git_remote(c)
    sync_pr(c)


def sync_with_git_remote(c: Context):
    echo_header("🚂 Syncing branch with remote")

    if not branch_exists_on_remote(c):
        c.run("git push --set-upstream origin HEAD")
    else:
        print("Pulling")
        c.run("git pull")
        print("Pushing")
        c.run("git push")


def sync_pr(c: Context):
    echo_header("💬 Syncing PR")
    # Get current branch name
    branch_name = Path(".git/HEAD").read_text().split("/")[-1].strip()
    pr_result: Result = c.run(
        "gh pr list --state OPEN",
        pty=False,
        hide=True,
    )

    if branch_name not in pr_result.stdout:
        create_pr(c)
    else:
        open_web = input("Open in browser? [y/n] ")
        if "y" in open_web.lower():
            c.run("gh pr view --web", pty=True)


def create_pr(c: Context):
    c.run(
        "gh pr create --web",
        pty=True,
    )


def branch_exists_on_remote(c: Context) -> bool:
    branch_name = Path(".git/HEAD").read_text().split("/")[-1].strip()

    branch_exists_result: Result = c.run(
        f"git ls-remote --heads origin {branch_name}",
        hide=True,
    )

    return branch_name in branch_exists_result.stdout


@task
def lint(c: Context):
    pre_commit(c)
    mypy(c)


def pre_commit(c: Context):
    echo_header("🧹 Running pre-commit checks")
    result = c.run("pre-commit run --all-files", pty=True, warn=True)

    if "fixed" in result.stdout or "reformatted" in result.stdout:
        add_commit(c, msg="style: linting")


def mypy(c: Context):
    echo_header("🧹 Running mypy")
    c.run("mypy .", pty=True)

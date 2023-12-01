import shutil
from pathlib import Path
from typing import Optional

from invoke import Context, task

from .logger import echo_header, msg_type

# Extract supported python versions from the pyproject.toml classifiers key
SUPPORTED_PYTHON_VERSIONS = [
    line.split("::")[-1].strip().replace('"', "").replace(",", "")
    for line in Path("pyproject.toml").read_text().splitlines()
    if "Programming Language :: Python ::" in line
]


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


def get_python_path(preferred_version: str) -> Optional[str]:
    """Get path to python executable."""
    preferred_version_path = shutil.which(f"python{preferred_version}")

    if preferred_version_path is not None:
        return preferred_version_path

    print(
        f"{msg_type.WARN}: python{preferred_version} not found, continuing with default python version",
    )
    return shutil.which("python")

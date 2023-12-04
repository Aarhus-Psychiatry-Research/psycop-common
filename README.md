# PSYCOP-COMMON

<!-- [![PyPI](https://img.shields.io/pypi/v/psycop-common.svg)][pypi status] -->
<!-- [![Python Version](https://img.shields.io/pypi/pyversions/psycop-common)][pypi status] -->
[![Open in Dev Container](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)][dev container]
[![Tests](https://github.com/Aarhus-Psychiatry-Research/psycop-common/actions/workflows/tests.yml/badge.svg)][tests]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
![python versions](https://img.shields.io/badge/Python=3.10-blue)

[dev container]: https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/Aarhus-Psychiatry-Research/psycop-common
[pypi status]: https://pypi.org/project/psycop-common/
[tests]: https://github.com/Aarhus-Psychiatry-Research/psycop-common/actions?workflow=Tests
[black]: https://github.com/psf/black



<!-- start short-description -->

The shared code across the PSYCOP projects.

<!-- end short-description -->

## Installation
### Method 1: Dev container
1. Install [Orbstack](https://orbstack.dev/) or Docker Desktop. Make sure to complete the full install process before continuing.
2. If not installed, install VSCode
3. Clone the repository and open it in VSCode
4. Press rebuild

![208704841-6bcefca0-9d76-48c5-b449-04d534340c4d](https://github.com/Aarhus-Psychiatry-Research/psycop-common/assets/8526086/d05d8dc7-cbb0-49ea-beb2-11d701440242)

5. Profit!

### Method 2: Ovartaci
If you're working on Ovartaci, all you need is to activate the shared environment by

```
conda activate psycop-main
```

To reinstall the packages in the environment, run `setup_overtaci.sh` in a `git bash` session.

### Method 3: Virtual environment install
If you're working locally:

```bash
git clone https://github.com/Aarhus-Psychiatry-Research/psycop-common.git
cd psycop-common

<activate virtual environment (conda, .venv etc.) here> 

# Install from the relevant requirements.txt files, e.g.
pip install -r requirements.txt -r dev-requirements.txt -r gpu-requirements.txt
```

<!--
To see more examples, see the [documentation].

# 📖 Documentation

| Documentation         |                                                          |
| --------------------- | -------------------------------------------------------- |
| 🔧 **[Installation]**  | Installation instructions on how to install this package |
| 📖 **[Documentation]** | A minimal and developing documentation                   |
| 👩‍💻 **[Tutorials]**     | Tutorials for using this package                         |
| 🎛️ **[API Reference]** | API reference for this package                           |
| 📚 **[FAQ]**           | Frequently asked questions                               |


# 💬 Where to ask questions

| Type                           |                        |
| ------------------------------ | ---------------------- |
| 📚 **FAQ**                      | [FAQ]                  |
| 🚨 **Bug Reports**              | [GitHub Issue Tracker] |
| 🎁 **Feature Requests & Ideas** | [GitHub Issue Tracker] |
| 👩‍💻 **Usage Questions**          | [GitHub Discussions]   |
| 🗯 **General Discussion**       | [GitHub Discussions]   |

[Documentation]: https://Aarhus-Psychiatry-Research.github.io/psycop-common/index.html
[Installation]: https://Aarhus-Psychiatry-Research.github.io/psycop-common/installation.html
[Tutorials]: https://Aarhus-Psychiatry-Research.github.io/psycop-common/tutorials.html
[API Reference]: https://Aarhus-Psychiatry-Research.github.io/psycop-common/references.html
[FAQ]: https://Aarhus-Psychiatry-Research.github.io/psycop-common/faq.html
[github issue tracker]: https://github.com/Aarhus-Psychiatry-Research/psycop-common/issues
[github discussions]: https://github.com/Aarhus-Psychiatry-Research/psycop-common/discussions
-->

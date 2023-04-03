# PSYCOP model evaluation

<!-- [![PyPI](https://img.shields.io/pypi/v/psycop-model-evaluation.svg)][pypi status] -->
<!-- [![Python Version](https://img.shields.io/pypi/pyversions/psycop-model-evaluation)][pypi status] -->
<!-- [![documentation](https://github.com/Aarhus-Psychiatry-Research/psycop-model-evaluation/workflows/documentation/badge.svg)][documentation] -->
<!-- [![Tests](https://github.com/Aarhus-Psychiatry-Research/psycop-model-evaluation/workflows/tests/badge.svg)][tests] -->
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

<!-- [pypi status]: https://pypi.org/project/psycop-model-evaluation/ -->
<!-- [documentation]: https://Aarhus-Psychiatry-Research.github.io/psycop-model-evaluation/ -->
[tests]: https://github.com/Aarhus-Psychiatry-Research/psycop-model-evaluation/actions?workflow=Tests
[black]: https://github.com/psf/black

Collection of model evaluations for the PSYCOP project.

The repository is structured like:
```
src/psycop_model_evaluation
├── base_charts.py                            # General charts, e.g. a generic "scatter" chart.
├── binary                                    # Evaluations of binary classification
│   ├── global_performance                    # Performance across subgroups and thresholds
│   │   ├── precision_recall.py
│   │   └── roc_auc.py
│   ├── performance_by_true_positive_rate.py  # Performance for given thresholds
│   ├── subgroups                             # Subgroups analyses
│   │   ├── age.py
│   │   ├── base.py
│   │   └── sex.py
│   ├── time                                  # Time analyses
│   │   ├── absolute_data.py                  # Absolute time is time as we usually think of it
│   │   ├── absolute_plots.py
│   │   ├── periodic_data.py                  # Periodic time is e.g. "day of week" or "week of year"
│   │   ├── periodic_plots.py
│   │   ├── timedelta_data.py                 # Timedelta is e.g. time from first visit
│   │   └── timedelta_plots.py
│   └── utils.py
├── descriptive_stats_table.py                # Descriptive statistics, e.g. a "table 1"
├── feature_importance                        # Feature importance
│   ├── feature_importance_table.py
│   └── sklearn
│       └── feature_importance.py
├── time_to_event                             # Evaluations for time to event
└── utils.py
```

## Installation

You can install _PSYCOP model evaluation_ into your `src` directory by:

```bash
pip install --src ./src -e git+https://github.com/Aarhus-Psychiatry-Research/psycop-model-evaluation#egg=psycop_model_evaluation
```

And when adding it to your dependencies in your `pyproject.toml`:

```bash
  "psycop-model-evaluation @ git+https://github.com/Aarhus-Psychiatry-Research/psycop-model-evaluation#egg=psycop_model_evaluation",
```

## Usage

For an example use case, see [t2d-baseline-paper](https://github.com/Aarhus-Psychiatry-Research/t2d-baseline-paper).

<!--
# 📖 Documentation

| Documentation         |                                                  |
| --------------------- | ------------------------------------------------ |
| 🔧 **[Installation]**  | Installation instructions for using this package |
| 📖 **[Documentation]** | A minimal and developing documentation           |
| 👩‍💻 **[Tutorials]**     | Tutorials for using this package                 |
| 🎛️ **[API Reference]** | API reference for this package                   |
| 📚 **[FAQ]**           | Frequently asked questions                       |
-->

# 💬 Where to ask questions

| Type                           |                        |
| ------------------------------ | ---------------------- |
| 📚 **FAQ**                      | [FAQ]                  |
| 🚨 **Bug Reports**              | [GitHub Issue Tracker] |
| 🎁 **Feature Requests & Ideas** | [GitHub Issue Tracker] |
| 👩‍💻 **Usage Questions**          | [GitHub Discussions]   |
| 🗯 **General Discussion**       | [GitHub Discussions]   |

<!-- [Documentation]: https://MartinBernstorff.github.io/psycop-model-evaluation/index.html
[Installation]: https://MartinBernstorff.github.io/psycop-model-evaluation/installation.html
[Tutorials]: https://MartinBernstorff.github.io/psycop-model-evaluation/tutorials.html
[API Reference]: https://MartinBernstorff.github.io/psycop-model-evaluation/references.html
[FAQ]: https://MartinBernstorff.github.io/psycop-model-evaluation/faq.html
[github issue tracker]: https://github.com/MartinBernstorff/psycop-model-evaluation/issues
[github discussions]: https://github.com/MartinBernstorff/psycop-model-evaluation/discussions
-->

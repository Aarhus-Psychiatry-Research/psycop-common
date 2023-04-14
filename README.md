Current priorities are on the 
[Board 🎬](https://github.com/orgs/Aarhus-Psychiatry-Research/projects/4/views/1).

psycop-model-training
==============================
![python versions](https://img.shields.io/badge/Python-%3E=3.9-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![Tests](https://github.com/MartinBernstorff/psycop-model-training/actions/workflows/tests.yml/badge.svg)][tests]

Prediction of type 2 diabetes among patients with visits to psychiatric hospital departments.

# Using the package
This is a set of modules used for some of the projects' model training. You need project-specific code to use these modules. To get started with that, see [template-model-training](https://github.com/Aarhus-Psychiatry-Research/template-model-training).

# Installing to src
```bash
pip install --src ./src -e git+https://github.com/Aarhus-Psychiatry-Research/psycop-model-training#egg=psycop_model_training
```

## Developing new evaluations
In general, model evaluations are added as their own file in 

```src > psycop_model_training > model_eval > base_artifacts > plots/tables```

To make sure they run every time, also add them to `base_artifact_generator.py`.

However, when developing, it's much faster to develop on a synthetic dataset.

To do that:

**Work locally**
1. Write your plot function in an appropriate file in the `src > psycop_model_training > model_eval > base_artifacts > plots/tables` directory. 
2. Test the plot on synthetic prediction data. Write a test in `tests > model_evaluation > test_name_of_your_plot.py`. 
    Use the other visualization tests as a guide.

**Work remotely**

3. When you're happy with the plot, test it on real data on Overtaci. To do this, go to Overtaci and replace the path in your test script with some real model predictions with metadata.

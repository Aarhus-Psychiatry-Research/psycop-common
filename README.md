Current priorities are on the 
[Board 🎬](https://github.com/orgs/Aarhus-Psychiatry-Research/projects/4/views/1).

psycop-model-training
==============================
![python versions](https://img.shields.io/badge/Python-%3E=3.7-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)

Prediction of type 2 diabetes among patients with visits to psychiatric hospital departments.

# Using the package
This is a set of modules used for some of the projects' model training. You need project-specific code to use these modules. To get started with that, see [template-model-training](https://github.com/Aarhus-Psychiatry-Research/template-model-training).

## Developing new evaluation plots
In general, model evaluations are added to `evaluate_model` in `psycop_model_training > evaluation.py`. However, when developing, it's much faster when you don't have to train a model for each iteration.

To do that:

**Work locally**
1. Write your plot function in an appropriate file in the `psycop_model_training > visualization` directory. 
2. Test the plot on synthetic prediction data. Write a test in `tests>name_of_your_plot.py`. 
    Use the `evaluate_saved_model_predictions.py` script or the other visualization tests as a guide.

**Work remotely**

3. When you're happy with the plot, test it on real data on Overtaci. To do this, go to Overtaci and replace the path in your test script with some real model predictions with metadata.
4. When all is ready to go, add the function to the `psycop_model_training > evaluation.py > evaluate_model` function



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

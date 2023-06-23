# Type 2 diabetes prediction
## Installation
All requirements are specified in the `pyproject.toml`. To install the project, clone it locally and run:

`pip install -e .`

To install the feature generation, model training and model evaluation dependencies that are shared across the PSYCOP projects in editable mode, run:

`pip install -r src-requirements.txt`

## Running the pipeline
1. Generate features using `src > t2d > feature_generation > main.py`

Please note that the feature generation pipeline is dependent on access the Central Denmark Region (CDR)'s SQL server. As such, it cannot be run outside the CDR network without approriate access.

2. Train models with `src > t2d > model_training > train_models_in_parallel.py`
3. Evaluate the models using `src > t2d > paper_outputs > [your_choice_of_eval_here]`

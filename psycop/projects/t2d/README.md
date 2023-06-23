# Type 2 diabetes prediction
## Installation
All requirements are specified in the `pyproject.toml`. To install the project, clone it locally and run:

`pip install -r dev-requirements`

## Running the pipeline
1. Generate features using `t2d > feature_generation > main.py`

Please note that the feature generation pipeline is dependent on access the Central Denmark Region (CDR)'s SQL server. As such, it cannot be run outside the CDR network without approriate access.

2. Train models with `t2d > model_training > train_models_in_parallel.py`
3. Evaluate the models using `t2d > paper_outputs > [your_choice_of_eval_here]`

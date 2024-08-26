# CVD prediction
## Installation
All requirements are specified in the `*requirements.txt` files. To install the project, clone it locally and run:

`pip install invoke`
`inv install-requirements`

## Running the pipeline
1. Generate features using `cvd > feature_generation > main.py`

Please note that the feature generation pipeline is dependent on access the Central Denmark Region (CDR)'s SQL server. As such, it cannot be run outside the CDR network without approriate access.

2. Train models with `cvd > model_training > main.py`
3. Evaluate the models using `cvd > paper_outputs > [your_choice_of_eval_here]`

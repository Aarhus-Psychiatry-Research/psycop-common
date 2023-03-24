"""Eval config schema."""
from psycop_model_training.config_schemas.basemodel import BaseModel


class EvalConfSchema(BaseModel):
    """Evaluation config."""

    force: bool = False
    # Whether to force evaluation even if wandb is not "run". Used for testing.

    descriptive_stats_table: bool = False
    # Whether to generate table 1.

    top_n_feature_importances: int
    # How many feature_importances to plot. Plots the most important n features. A table with all features is also logged.

    positive_rates: list[int]
    # DEPRECATED, not removed to avoid breaking runs. We'll soon deprecate the entire eval part of psycop-model-training and move it to psycop-model-eval: The threshold mapping a model's predicted probability to a binary outcome can be computed if we know, which positive rate we're targeting. We can't know beforehand which positive rate is best, because it's a trade-off between false-positives and false-negatives. Therefore, we compute performacne for a range of positive rates.

    save_model_predictions_on_overtaci: bool

    lookahead_bins: list[int]
    # List of lookahead distances for plotting. Will create bins in between each distances. E.g. if specifying 1, 5, 10, will bin evaluation as follows: [0, 1], [1, 5], [5, 10], [10, inf].

    lookbehind_bins: list[int]
    # List of lookbehidn distances for plotting. Will create bins in between each distances. E.g. if specifying 1, 5, 10, will bin evaluation as follows: [0, 1], [1, 5], [5, 10], [10, inf].

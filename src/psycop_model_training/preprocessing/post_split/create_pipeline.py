"""Create preprocessing pipeline based on config."""
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from sklearn.feature_selection import (
    SelectPercentile,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wasabi import Printer


def get_feature_selection_steps(cfg: FullConfigSchema) -> list:
    """Add feature selection steps to the preprocessing pipeline."""
    new_steps = []

    if cfg.preprocessing.post_split.feature_selection.name:
        if cfg.preprocessing.post_split.feature_selection.name == "f_classif":
            new_steps.append(
                (
                    "feature_selection",
                    SelectPercentile(
                        f_classif,
                        percentile=cfg.preprocessing.post_split.feature_selection.params[  # type: ignore
                            "percentile"
                        ],
                    ),
                ),
            )
        elif cfg.preprocessing.post_split.feature_selection.name == "chi2":
            new_steps.append(
                (
                    "feature_selection",
                    SelectPercentile(
                        chi2,
                        percentile=cfg.preprocessing.post_split.feature_selection.params[  # type: ignore
                            "percentile"
                        ],
                    ),
                ),
            )
        elif (
            cfg.preprocessing.post_split.feature_selection.name == "mutual_info_classif"
        ):
            new_steps.append(
                (
                    "feature_selection",
                    SelectPercentile(
                        mutual_info_classif,
                        percentile=cfg.preprocessing.post_split.feature_selection.params[  # type: ignore
                            "percentile"
                        ],
                    ),
                ),
            )
        else:
            raise ValueError(
                f"Unknown feature selection method {cfg.preprocessing.post_split.feature_selection.name}",
            )

    return new_steps


def create_preprocessing_pipeline(cfg: FullConfigSchema) -> Pipeline:
    """Create preprocessing pipeline based on config."""
    msg = Printer(timestamp=True)

    steps = []
    # Imputation
    if (
        cfg.model.require_imputation
        and not cfg.preprocessing.post_split.imputation_method
    ):
        msg.warn(
            f"{cfg.model.name} requires imputation, but no imputation method was specified in the config file. Overriding to 'mean'.",
        )

        steps.append(
            (
                "Imputation",
                SimpleImputer(strategy="mean"),
            ),
        )
        # Not a great solution, but preferable to the script breaking and stopping a hyperparameter search.

    if cfg.preprocessing.post_split.imputation_method:
        steps.append(
            (
                "Imputation",
                SimpleImputer(strategy=cfg.preprocessing.post_split.imputation_method),
            ),
        )

    # Feature selection
    # Important to do this before scaling, since chi2
    # requires non-negative values
    steps += get_feature_selection_steps(cfg)

    # Feature scaling
    # Important to do this after feature selection, since
    # half of the values in z-score normalisation will be negative,
    # which is not allowed for chi2
    if cfg.preprocessing.post_split.scaling:
        if cfg.preprocessing.post_split.scaling in {
            "z-score-normalization",
            "z-score-normalisation",
        }:
            steps.append(
                ("z-score-normalization", StandardScaler()),
            )
        else:
            raise ValueError(
                f"{cfg.preprocessing.post_split.scaling} is not implemented. See above",
            )

    return Pipeline(steps)

"""Create preprocessing pipeline based on config."""
from sklearn.feature_selection import (
    SelectPercentile,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from psycop_model_training.config.schemas import FullConfigSchema
from psycop_model_training.preprocessing.feature_selectors import DropDateTimeColumns
from psycop_model_training.preprocessing.feature_transformers import (
    ConvertToBoolean,
    DateTimeConverter,
)


def get_feature_selection_steps(cfg):
    """Add feature selection steps to the preprocessing pipeline."""
    new_steps = []

    if cfg.preprocessing.feature_selection.name:
        if cfg.preprocessing.feature_selection.name == "f_classif":
            new_steps.append(
                (
                    "feature_selection",
                    SelectPercentile(
                        f_classif,
                        percentile=cfg.preprocessing.feature_selection.params[
                            "percentile"
                        ],
                    ),
                ),
            )
        elif cfg.preprocessing.feature_selection.name == "chi2":
            new_steps.append(
                (
                    "feature_selection",
                    SelectPercentile(
                        chi2,
                        percentile=cfg.preprocessing.feature_selection.params[
                            "percentile"
                        ],
                    ),
                ),
            )
        elif cfg.preprocessing.feature_selection.name == "mutual_info_classif":
            new_steps.append(
                (
                    "feature_selection",
                    SelectPercentile(
                        mutual_info_classif,
                        percentile=cfg.preprocessing.feature_selection.params[
                            "percentile"
                        ],
                    ),
                ),
            )
        else:
            raise ValueError(
                f"Unknown feature selection method {cfg.preprocessing.feature_selection.name}",
            )

    return new_steps


def create_preprocessing_pipeline(cfg: FullConfigSchema):
    """Create preprocessing pipeline based on config."""
    steps = []
    # Conversion
    if cfg.preprocessing.drop_datetime_predictor_columns:
        steps.append(
            (
                "DropDateTimeColumns",
                DropDateTimeColumns(pred_prefix=cfg.data.pred_prefix),
            ),
        )

    if cfg.preprocessing.convert_datetimes_to_ordinal:
        dtconverter = DateTimeConverter()
        steps.append(("DateTimeConverter", dtconverter))

    if cfg.preprocessing.convert_to_boolean:
        steps.append(("ConvertToBoolean", ConvertToBoolean()))

    # Imputation
    if cfg.model.require_imputation and not cfg.preprocessing.imputation_method:
        raise ValueError(
            f"{cfg.model.name} requires imputation, but no imputation method was specified in the config file.",
        )

    if cfg.preprocessing.imputation_method:
        steps.append(
            (
                "Imputation",
                SimpleImputer(strategy=cfg.preprocessing.imputation_method),
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
    if cfg.preprocessing.scaling:
        if cfg.preprocessing.scaling in {
            "z-score-normalization",
            "z-score-normalisation",
        }:
            steps.append(
                ("z-score-normalization", StandardScaler()),
            )
        else:
            raise ValueError(
                f"{cfg.preprocessing.scaling} is not implemented. See above",
            )

    return Pipeline(steps)

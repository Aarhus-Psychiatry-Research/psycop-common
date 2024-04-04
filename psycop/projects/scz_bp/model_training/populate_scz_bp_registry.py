# ruff: noqa


def populate_scz_bp_registry():
    from psycop.projects.scz_bp.model_training.binary_classification_pipeline.pipeline_constructor import (
        imblearn_pipeline_constructor,
    )
    from psycop.projects.scz_bp.model_training.estimator_steps.standard_scaler import (
        StandardScaler,
        standard_scaler_step,
    )
    from psycop.projects.scz_bp.model_training.estimator_steps.synth_data_augmentation import (
        SynthcityAugmentationSuggester,
        SyntheticDataAugmentation,
        synthetic_data_augmentation_step,
    )

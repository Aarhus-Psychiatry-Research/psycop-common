# ruff: noqa


def populate_scz_bp_registry() -> None:
    from psycop.projects.scz_bp.model_training.estimator_steps.cleanlab_processing import (
        CleanlabProcessing,
    )
    from psycop.projects.scz_bp.model_training.estimator_steps.imputers import (
        miss_forest_imputation_step,
        simple_imputation_step,
    )
    from psycop.projects.scz_bp.model_training.estimator_steps.synth_data_augmentation import (
        SyntheticDataAugmentation,
    )

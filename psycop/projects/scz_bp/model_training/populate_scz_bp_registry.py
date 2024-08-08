# ruff: noqa


def populate_scz_bp_registry():
    from psycop.projects.scz_bp.model_training.binary_classification_pipeline.pipeline_constructor import (
        imblearn_pipeline_constructor,
    )
    from psycop.projects.scz_bp.model_training.estimator_steps.smote import smote_step
    from psycop.projects.scz_bp.model_training.estimator_steps.standard_scaler import (
        StandardScaler,
        standard_scaler_step,
    )
    from psycop.projects.scz_bp.model_training.synthetic_trainer.synthetic_cv_trainer import (
        SyntheticCrossValidatorTrainer,
    )
    from psycop.projects.scz_bp.model_training.synthetic_trainer.synthetic_data_loader import (
        SyntheticVerticalConcatenator,
    )
    from psycop.projects.scz_bp.model_training.synthetic_trainer.synthetic_split_trainer import (
        SyntheticSplitTrainerSeparatePreprocessing,
    )

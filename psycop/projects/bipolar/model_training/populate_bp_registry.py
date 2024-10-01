# ruff: noqa


def populate_with_bp_registry() -> None:
    from psycop.common.model_training_v2.trainer.data.dataloaders import ParquetVerticalConcatenator
    from psycop.common.model_training_v2.trainer.preprocessing.steps.cell_transformers import (
        BoolToInt,
    )
    from psycop.common.model_training_v2.trainer.preprocessing.steps.column_filters import (
        RegexColumnBlacklist,
        TemporalColumnFilter,
    )


populate_with_bp_registry()

# ruff: noqa


def populate_with_restraint_registry() -> None:
    from psycop.common.model_training_v2.trainer.data.dataloaders import ParquetVerticalConcatenator
    from psycop.common.model_training_v2.trainer.preprocessing.steps.column_filters import (
        RegexColumnBlacklist,
    )
    from psycop.common.model_training_v2.trainer.preprocessing.steps.column_filters import (
        TemporalColumnFilter,
    )
    from psycop.common.model_training_v2.trainer.preprocessing.steps.cell_transformers import (
        BoolToInt,
    )
    from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
        RegionalFilter,
    )


populate_with_restraint_registry()

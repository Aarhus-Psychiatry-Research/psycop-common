# ruff: noqa


def populate_with_cvd_registry() -> None:
    from psycop.projects.cvd.model_training.data_loader.trainval_loader import (
        ParquetVerticalConcatenator,
    )
    from psycop.common.model_training_v2.trainer.preprocessing.steps.column_filters import (
        RegexColumnBlacklist,
    )
    from psycop.projects.cvd.model_training.preprocessing.datetime_filter import (
        TemporalColumnFilter,
    )
    from psycop.projects.cvd.model_training.preprocessing.bool_to_int import BoolToInt


populate_with_cvd_registry()

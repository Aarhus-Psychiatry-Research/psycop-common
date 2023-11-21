# ruff: noqa


def populate_with_cvd_registry() -> None:
    from psycop.projects.cvd.model_training.data_loader.trainval_loader import (
        ParquetVerticalConcatenator,
    )
    from psycop.projects.cvd.model_training.preprocessing.regex_filter import (
        RegexColumnBlacklist,
    )
    from psycop.common.model_training_v2.trainer.preprocessing.steps.col_filters import (
        TemporalColumnFilter,
    )
    from psycop.projects.cvd.model_training.preprocessing.bool_to_int import BoolToInt


populate_with_cvd_registry()

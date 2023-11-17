# ruff: noqa

def populate_with_cvd_registry() -> None:
    from psycop.projects.cvd.model_training.data_loader.trainval_loader import ParquetVerticalConcatenator

populate_with_cvd_registry()
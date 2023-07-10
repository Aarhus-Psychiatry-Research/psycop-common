from pathlib import Path

import polars as pl

from psycop.common.patient_print.healthprints_config import (
    HEALTHPRINT_PLOTS_DIR,
    HEALTHPRINTS_DATASETS_DIR,
)
from psycop.common.patient_print.test_patient_printer import (
    HealthPrintPredictionTime,
    create_health_prints_from_patients,
)


def create_plots_from_dataset(
    dataset_path: Path,
    subtype: str,
    y_min: float,
    y_max: float,
    output_dir: Path,
):
    dataset = pl.read_parquet(dataset_path)

    massaged_ds = (
        dataset.with_columns(
            (
                (pl.col("event_timestamp") - pl.col("pred_timestamp")).dt.minutes()
                / (24 * 60)
            )
            .round(1)
            .alias("rel_time")
        )
        .select(["pred_time_uuid", "event_type", "event_value", "rel_time"])
        .rename({"event_type": "type", "event_value": "value"})
    )

    patient_dicts = massaged_ds.partition_by("pred_time_uuid", as_dict=True)

    patients = [
        HealthPrintPredictionTime(
            df=patient_dicts[k],
            i=i,
            x_max=-0,
            x_min=(-365 * 5),
            y_min=y_min,
            y_max=y_max,
            subtype=subtype,
            output_dir=output_dir,
        )
        for i, k in enumerate(patient_dicts)
    ]

    create_health_prints_from_patients(patients=patients)


if __name__ == "__main__":
    subtypes = ("negative", "positive")
    assert all(
        (HEALTHPRINTS_DATASETS_DIR / f"{subtype}.parquet").exists()
        for subtype in subtypes
    )

    combined_ds = pl.concat(
        [
            pl.read_parquet(HEALTHPRINTS_DATASETS_DIR / f"{subtype}.parquet")
            for subtype in subtypes
        ]
    )
    y_min: float = combined_ds.get_column("event_value").min()  # type: ignore
    y_max: float = combined_ds.get_column("event_value").max()  # type: ignore

    for subtype in subtypes:
        create_plots_from_dataset(
            dataset_path=HEALTHPRINTS_DATASETS_DIR / f"{subtype}.parquet",
            subtype=subtype,
            y_min=y_min,
            y_max=y_max,
            output_dir=Path(HEALTHPRINT_PLOTS_DIR),
        )

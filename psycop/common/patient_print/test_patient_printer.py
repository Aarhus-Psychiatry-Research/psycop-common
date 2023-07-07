import time
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import polars as pl

from psycop.common.test_utils.str_to_df import str_to_pl_df

mplstyle.use("fast")
import multiprocessing
from pathlib import Path

import pytest
import tqdm


@dataclass(frozen=True)
class HealthPrintPredictionTime:
    df: pl.DataFrame
    i: int
    subtype: str
    output_dir: Path
    x_min: float
    x_max: float
    x_axis: str = "rel_time"
    y_axis: str = "type"
    color: str = "value"


@dataclass(frozen=True)
class PatientType:
    diabetes: bool
    data: pl.DataFrame


diabetes_patient = PatientType(
    diabetes=True,
    data=str_to_pl_df(
        """id,type,rel_time,value,
            1,weight,-25,1,
            1,weight,-16,1.2,
            1,BMI,-8,0.8,
            1,BMI,-10,1,
            1,HbA1c,-1,1,
            1,HbA1c,-28,1.3,
        """
    ),
)

no_diabetes_patient = PatientType(
    diabetes=False,
    data=str_to_pl_df(
        """id,type,rel_time,value,
            1,weight,-1,1,
            1,weight,-5,0.9,
            1,BMI,-1,1.2,
            1,BMI,-3,0.8,
            1,HbA1c,-1,1,
            1,HbA1c,-24,0.9,
        """
    ),
)


def save_health_print(
    data: HealthPrintPredictionTime,
):
    df = data.df
    plt.scatter(df[data.x_axis], df[data.y_axis], c=df[data.color])
    plt.xlim(data.x_min, data.x_max)

    data.output_dir.mkdir(exist_ok=True, parents=True)

    plt.savefig(f"{data.output_dir}/{data.subtype}_{data.i}.jpg")
    plt.close()


@pytest.mark.parametrize(
    "n_plots",
    [1_000],
)
@pytest.mark.parametrize(
    "patient",
    [diabetes_patient, no_diabetes_patient],
)
def test_patient_print(n_plots: int, patient: PatientType, tmp_path: Path):
    dataframe_containers = [
        HealthPrintPredictionTime(
            df=patient.data,
            i=i,
            x_min=0,
            x_max=-365,
            subtype="diabetes" if patient.diabetes else "control",
            output_dir=tmp_path,
        )
        for i in range(n_plots)
    ]

    create_health_prints_from_patients(dataframe_containers)


def create_health_prints_from_patients(patients: Sequence[HealthPrintPredictionTime]):
    # Map dataframes to plotting function
    chunk_size = (len(patients) // (multiprocessing.cpu_count())) * 2

    with multiprocessing.Pool() as pool:
        var = list(
            tqdm.tqdm(
                pool.imap(
                    save_health_print,
                    patients,
                    chunk_size,
                ),
                total=len(patients),
            )
        )

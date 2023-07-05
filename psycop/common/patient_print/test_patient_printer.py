import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import polars as pl

from psycop.common.test_utils.str_to_df import str_to_pl_df

mplstyle.use("fast")
import multiprocessing
from pathlib import Path

import pytest


@dataclass(frozen=True)
class DataframeContainer:
    df: pl.DataFrame
    i: int
    folder: str


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


def plot(data: DataframeContainer):
    df = data.df
    plt.scatter(df["rel_time"], df["type"], c=df["value"])

    folder = Path("plots") / data.folder
    folder.mkdir(exist_ok=True, parents=True)

    plt.savefig(f"{folder}/test_{data.i}.jpg")
    plt.close()


@pytest.mark.parametrize(
    "n_plots",
    [1_000],
)
@pytest.mark.parametrize(
    "patient",
    [diabetes_patient, no_diabetes_patient],
)
def test_patient_print(n_plots: int, patient: PatientType):
    dataframe_containers = [
        DataframeContainer(
            df=patient.data,
            i=i,
            folder="diabetes" if patient.diabetes else "no_diabetes",
        )
        for i in range(n_plots)
    ]

    # Map dataframes to plotting function
    start_time = time.time()

    chunk_size = len(dataframe_containers) // (multiprocessing.cpu_count() * 2)

    with multiprocessing.Pool() as pool:
        pool.map(plot, dataframe_containers, chunk_size)

    end_time = time.time()

    # Write time taken to benchmark.txt
    with Path("benchmark.txt").open("a") as f:
        f.write(
            f"Time taken to plot {n_plots} plots: {end_time - start_time} seconds\n"
        )

    pass

import numpy as np
import pandas as pd
import patchworklib as pw
import plotnine as pn

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR

data_path=OVARTACI_SHARED_DIR / "uti" / "flattened_datasets" / "flattened_datasets"

full_definition = pd.read_parquet(data_path / 'uti_outcomes_full_definition' / 'uti_outcomes_full_definition.parquet')
urine_samples = pd.read_parquet(data_path / 'uti_outcomes_urine_samples' / 'uti_outcomes_urine_samples.parquet')
antibiotics = pd.read_parquet(data_path / 'uti_outcomes_antibiotics' / 'uti_outcomes_antibiotics.parquet')

np.random.seed(42)

# Parameters for the synthetic dataset
n_rows = 2000  # Number of rows
patient_ids = np.random.randint(1, 101, size=n_rows)  # 100 unique patients
timestamps = pd.date_range(start='2011-01-01', end='2021-12-31', periods=n_rows)
values = np.random.choice([0, 6600, 6601, 6001, 6002], size=n_rows, p=[0.5, 0.125, 0.125, 0.125, 0.125])

data = pd.DataFrame({
    'timestamp': timestamps,
    'id': patient_ids,
    'value': values
})

def plot_prediction_times_by_year(data: pd.DataFrame):
    pass
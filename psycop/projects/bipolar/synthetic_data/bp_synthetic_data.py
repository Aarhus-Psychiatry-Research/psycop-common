from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def bp_synthetic_data(num_patients: int, min_rows: int = 3, max_rows: int = 20) -> pd.DataFrame:
    patient_ids = []
    timestamps = []
    data = []

    for patient_id in range(1, num_patients + 1):
        num_rows = np.random.randint(min_rows, max_rows + 1)

        # sample random start date month between 2010 and 2012
        start_date = datetime(
            year=np.random.randint(2010, 2012), month=np.random.randint(1, 13), day=1
        )

        for _ in range(num_rows):
            patient_ids.append(patient_id)
            timestamps.append(start_date)

            start_date += relativedelta(months=1)

            # Generate random data for each column
            row_data = np.random.rand(50)
            data.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(data, columns=[f"pred_{i+1}" for i in range(50)])
    df["patient_id"] = patient_ids
    df["timestamp"] = timestamps

    return df


def bp_synthetic_eval_data(
    num_patients: int, min_rows: int = 3, max_rows: int = 20
) -> pd.DataFrame:
    synthetic_df = bp_synthetic_data(num_patients, min_rows, max_rows)

    # Add a label column
    synthetic_df["label"] = np.random.randint(0, 2, size=synthetic_df.shape[0])

    # Add a y_pred column (float between 0 and 1)
    synthetic_df["y_pred"] = np.random.rand(synthetic_df.shape[0])

    # Add a predicted label column (0 or 1), where 0 if y_pred < 0.5, 1 otherwise
    synthetic_df["predicted_label"] = (synthetic_df["y_pred"] >= 0.5).astype(int)

    return synthetic_df


if __name__ == "__main__":
    # Generate synthetic data
    synthetic_eval_df = bp_synthetic_eval_data(num_patients=50, min_rows=10, max_rows=30)

    # Display first few rows of the DataFrame
    print(synthetic_eval_df.head())

"""Example of how to inspect the dataset."""
from datetime import datetime
from pathlib import Path

from psycopt2d.load import load_dataset


def main():
    """Main function."""
    dataset_dir = Path(
        "E:/shared_resources/feature_sets/t2d/feature_sets/psycop_t2d_adminmanber_201_features_2022_10_05_15_14",
    )

    # Load the dataset
    dataset = load_dataset(
        split_names="train",
        dir_path=dataset_dir,
        min_lookahead_days=1825,
        drop_patient_if_outcome_before_date=datetime(2013, 1, 1),
        file_suffix="parquet",
    )

    return dataset


if __name__ == "__main__":
    main()

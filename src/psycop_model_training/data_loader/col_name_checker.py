"""Check that all columns in the config exist in the dataset."""
import pandas as pd

from psycop_model_training.utils.config_schemas.data import ColumnNamesSchema


def check_columns_exist_in_dataset(
    col_name_schema: ColumnNamesSchema, df: pd.DataFrame
):
    """Check that all columns in the config exist in the dataset."""
    # Iterate over attributes in the config
    missing_columns = []

    for attr in dir(col_name_schema):
        # Skip private attributes
        if attr.startswith("_"):
            continue

        # Skip col names that are not string
        if not isinstance(getattr(col_name_schema, attr), str):
            continue

        # Check that the column exists in the dataset
        if not getattr(col_name_schema, attr) in df:
            missing_columns.append(getattr(col_name_schema, attr))

    if missing_columns:
        raise ValueError(
            f"Columns in config but not in dataset: {missing_columns}. Columns in dataset: {df.columns}"
        )

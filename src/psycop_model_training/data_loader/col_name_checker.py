"""Check that all columns in the config exist in the dataset."""

import Levenshtein
import pandas as pd
from psycop_model_training.config_schemas.data import ColumnNamesSchema


def get_most_likely_str_from_edit_distance(
    candidate_strs: list[str],
    input_str: str,
    n_str_to_return: int,
    edit_distance_threshold: int = 15,
) -> list[str]:
    """Get most likely string from edit distance.

    Args:
        candidate_strs (list[str]): List of candidate strings.
        input_str (str): The incorrect string.
        n_str_to_return (int): Number of strings to return.
        edit_distance_threshold (int, optional): Maximum edit distance to consider. Defaults to 5.

    Returns:
        str: String from candidate_strs that is most similar to input_str by edit distance.
    """
    # Compute the Levenshtein distance between the input string and each candidate string.
    distances = [
        Levenshtein.distance(input_str, candidate) for candidate in candidate_strs
    ]

    # Sort the candidate strings by their Levenshtein distance from the input string.
    sorted_candidates = [
        x
        for distance, x in sorted(zip(distances, candidate_strs))
        if distance <= edit_distance_threshold
    ]

    # Return the first `n_str_to_return` elements of the sorted list of candidate strings.
    return sorted_candidates[:n_str_to_return]


def check_columns_exist_in_dataset(
    col_name_schema: ColumnNamesSchema,
    df: pd.DataFrame,
):
    """Check that all columns in the config exist in the dataset."""
    # Iterate over attributes in the config
    error_strs = []
    col_names = []

    # Get the column names to check
    for attr in dir(col_name_schema):
        # Skip private attributes
        if attr.startswith("_"):
            continue

        attr_val = getattr(col_name_schema, attr)

        if isinstance(attr_val, str):
            col_names.append(attr_val)
            continue

        # Skip col names that are not string
        if not isinstance(attr_val, str) and isinstance(attr_val, list):
            for item in attr_val:
                if isinstance(item, str):
                    col_names.append(item)

    # Check that the col_names exist in the dataset
    for item in col_names:
        # Check that the column exists in the dataset
        if item not in df:
            most_likely_alternatives = get_most_likely_str_from_edit_distance(
                candidate_strs=list(df.columns),
                input_str=item,
                n_str_to_return=3,
            )

            error_str = f"Column '{item}' in config but not in dataset.\n"
            error_str += f"    Did you mean {most_likely_alternatives}? \n"
            error_strs.append(error_str)

    if error_strs:
        raise ValueError("\n".join(error_strs))

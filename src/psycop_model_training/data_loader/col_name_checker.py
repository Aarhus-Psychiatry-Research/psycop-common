"""Check that all columns in the config exist in the dataset."""
from typing import List

import Levenshtein
import pandas as pd

from psycop_model_training.utils.config_schemas.data import ColumnNamesSchema


def check_columns_exist_in_dataset(
    col_name_schema: ColumnNamesSchema, df: pd.DataFrame
):
    """Check that all columns in the config exist in the dataset."""
    # Iterate over attributes in the config
    error_strs = []

    for attr in dir(col_name_schema):
        # Skip private attributes
        if attr.startswith("_"):
            continue

        col_name = getattr(col_name_schema, attr)

        # Skip col names that are not string
        if not isinstance(col_name, str):
            continue

        # Check that the column exists in the dataset
        if not col_name in df:
            most_likely_alternatives = get_most_likely_str_from_edit_distance(
                candidate_strs=df.columns,
                input_str=col_name,
                n_str_to_return=3,
            )

            error_str = f"Column '{col_name}' in config but not in dataset.\n"
            error_str += f"    Did you mean {most_likely_alternatives}? \n"
            error_strs.append(error_str)

    if error_strs:
        raise ValueError("\n".join(error_strs))


def get_most_likely_str_from_edit_distance(
    candidate_strs: list[str],
    input_str: str,
    n_str_to_return: int,
    edit_distance_threshold: int = 15,
) -> List[str]:
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

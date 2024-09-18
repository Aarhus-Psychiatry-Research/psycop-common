"""Check that all feature_dicts conform to correct formatting.

Also kcheck that they return meaningful dictionaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wasabi import Printer

from psycop.common.feature_generation.data_checks.raw.check_raw_df import check_raw_df

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def check_df_conforms_to_feature_spec(
    df: pd.DataFrame,
    msg_prefix: str,
    arg_dict: dict,  # type: ignore
    allowed_nan_value_prop: float = 0.01,
    required_columns: Sequence[str] = ("dw_ek_borger", "timestamp", "value"),
    subset_duplicates_columns: Sequence[str] = ("dw_ek_borger", "timestamp", "value"),
    expected_val_dtypes: Sequence[str] = ("float64", "int64"),
) -> dict[Any, list[str]] | None:
    """Check that a loaded df conforms to to a given feature specification.
    Useful when creating loaders.

    Args:
        df (pd.DataFrame): Dataframe to check.
        subset_duplicates_columns (list[str]): list of columns to subset on when
            checking for duplicates.
        expected_val_dtypes (list[str]): Expected value dtype.
        msg_prefix (str): Prefix to add to all messages.
        arg_dict (dict): Dictionary with specifications for what df should conform to.
        allowed_nan_value_prop (float): Allowed proportion of missing values. Defaults to 0.01.
        required_columns (Sequence[str]): Sequence of columns that must exist.

    Raises:
        ValueError: If df does not conform to d.
    """

    msg = Printer(timestamp=True)

    allowed_nan_value_prop = (
        arg_dict["allowed_nan_value_prop"]
        if arg_dict["allowed_nan_value_prop"]
        else allowed_nan_value_prop
    )

    source_failures, _ = check_raw_df(
        df=df,
        required_columns=required_columns,
        subset_duplicates_columns=subset_duplicates_columns,
        expected_val_dtypes=expected_val_dtypes,
    )

    # Return errors
    if len(source_failures) != 0:
        msg.fail(f"{msg_prefix} errors: {source_failures}")
        return {arg_dict["predictor_df"]: source_failures}

    msg.good(f"{msg_prefix} passed data validation criteria.")
    return None


def get_unique_predictor_dfs(
    predictor_dict_list: Sequence[dict],  # type: ignore
    required_keys: list[str],
) -> list[Any]:
    """Get unique predictor_dfs from predictor_dict_list.

    Args:
        predictor_dict_list (list[dict]): list of dictionaries where the key predictor_df maps to a catalogue registered data loader
            or is a valid dataframe.
        required_keys (list[str]): list of required keys.

    Returns:
        list[dict]: list of unique predictor_dfs.
    """

    dicts_with_subset_keys = []

    for d in predictor_dict_list:
        new_d = {k: d[k] for k in required_keys}

        if "loader_kwargs" in d:
            new_d["loader_kwargs"] = d["loader_kwargs"]

        dicts_with_subset_keys.append(new_d)

    unique_subset_dicts = []

    for predictor_dict in dicts_with_subset_keys:
        if predictor_dict not in unique_subset_dicts:
            unique_subset_dicts.append(predictor_dict)

    return unique_subset_dicts

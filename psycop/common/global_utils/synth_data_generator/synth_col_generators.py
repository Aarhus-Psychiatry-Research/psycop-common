"""Column generators for synthetic data."""

from collections.abc import Iterable
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from transformers import PreTrainedModel, PreTrainedTokenizer  # type: ignore

from psycop.common.global_utils.paths import PSYCOP_PKG_ROOT


def create_outcome_values(
    df: pd.DataFrame,
    n_samples: int,
    logistic_outcome_model: str,
    intercept: float = 0,
    noise_mean_sd: tuple[float, float] = (0, 1),
) -> pd.Series:  # type: ignore
    """Create outcome values for a column.

    Args:
        df (pd.DataFrame): The dataframe to base the outcome values on. Should contain all the columns used in the logistic_outcome_model.
        n_samples (int): Number of samples (rows) to generate.
        logistic_outcome_model (str): The statistical model used to generate outcome values, e.g. specified as'1*col_name+1*col_name2'.
        intercept (float, optional): The intercept of the logistic outcome model. Defaults to 0.
        noise_mean_sd (tuple[float, float], optional): Mean and sd of the noise.
            Increase SD to obtain more uncertain models.

    Returns:
        pd.Series: The outcome values.
    """
    # Linear model with columns
    _y = intercept

    for var in logistic_outcome_model.split("+"):
        effect, col = var.split("*")
        _y = float(effect) * df[col] + _y

    noise = np.random.normal(loc=noise_mean_sd[0], scale=noise_mean_sd[1], size=n_samples)

    # Z-score normalise and add noise
    _y = stats.zscore(_y) + noise

    out = 1 / (1 + np.exp(_y))  # type: ignore
    return out  # type: ignore


def generate_text_data(
    n_samples: int,
    sequence: str,
    tokenizer: PreTrainedTokenizer | None = None,
    model: PreTrainedModel | None = None,
) -> list[str]:
    """Generate text data.

    Args:
        n_samples (int): Number of rows to generate.
        sequence (str): Text prompt to use for generating text data. Defaults to "The quick brown fox jumps over the lazy dog".
        tokenizer (Optional[Any]): Huggingface tokenizer
        model (Optional[Any]): Huggingface model

    Returns:
        list[str]: list of generated text data.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer  # type: ignore

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2") if tokenizer is None else tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2") if model is None else model  # type: ignore

    inputs = tokenizer.encode(sequence, return_tensors="pt")

    generated_texts = []
    for _ in range(n_samples):
        max_tokens = np.random.randint(low=0, high=500, size=1)[0]

        outputs = model.generate(  # type: ignore
            inputs,  # type: ignore
            min_length=0,
            max_length=max_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(text)

    return generated_texts


def generate_col_from_specs(
    column_type: str,
    n_samples: int,
    col_specs: dict,  # type: ignore
    sequence: Optional[str],
    tokenizer: Optional[Any] = None,
    model: Optional[Any] = None,
) -> Iterable:  # type: ignore
    """Generate a column of data.

    Args:
        column_type (str): Type of column to generate. Either uniform_int, text, id or datetime_uniform.
        n_samples (int): Number of rows to generate.
        col_specs (dict): A dict representing each column. Key is col_name (str), values is a dict with column_type (str), min (int) and max(int).
        sequence (str): Text prompt to use for generating text data. Defaults to "The quick brown fox jumps over the lazy dog".
        tokenizer (Optional[Any]): Huggingface tokenizer.
        model (Optional[Any]): Huggingface model.

    Raises:
        ValueError: If column_type isn't either uniform_int, text, or datetime_uniform.

    Returns:
        Iterable: The generated column.
    """

    if column_type == "text":
        if sequence is None:
            raise ValueError("If column_type is text, sequence must be specified.")
        generated_texts = generate_text_data(
            n_samples=n_samples, sequence=sequence, tokenizer=tokenizer, model=model
        )

        return generated_texts

    if column_type == "id":
        return -np.arange(n_samples)

    if column_type == "uniform_int":
        return np.random.randint(low=col_specs["min"], high=col_specs["max"], size=n_samples)

    if column_type == "uniform_float":
        return np.random.uniform(low=col_specs["min"], high=col_specs["max"], size=n_samples)

    if column_type == "normal":
        return np.random.normal(loc=col_specs["mean"], scale=col_specs["sd"], size=n_samples)

    if column_type == "datetime_uniform":
        return pd.to_datetime(
            np.random.uniform(  # type: ignore
                low=col_specs["min"], high=col_specs["max"], size=n_samples
            ),
            unit="D",
        ).round(  # type: ignore
            "min"
        )

    raise ValueError(f"Unknown distribution: {column_type}")


def generate_data_columns(
    predictors: Iterable[dict],  # type: ignore
    n_samples: int,
    df: pd.DataFrame = pd.DataFrame(),  # noqa: B008
    text_prompt: Optional[str] = None,
) -> pd.DataFrame:
    """Generate a dataframe with columns from the predictors iterable.

    Args:
        predictors (iter[dict]): A dict representing each column. Key is col_name (str), values is a dict with column_type (str), min (int) and max(int).
        n_samples (int): Number of rows to generate.
        df (pd.DataFrame): Dataframe to append to.
        text_prompt (str): Text prompt to use for generating text data. Defaults to "The quick brown fox jumps over the lazy dog".

    Raises:
        ValueError: If column_type isn't either uniform_int, text, or datetime_uniform.

    Returns:
        pd.DataFrame: The generated dataframe.


    Example:
        >>> column_specifications = {
        >>>   "citizen_ids": {"column_type": "uniform_int", "min": 0, "max": 1_200_000},
        >>>   "timestamp": {"column_type": "datetime_uniform", "min": 0, "max": 5 * 365},
        >>>   "text": {"column_type": "text"},
        >>> }
        >>>
        >>> df = generate_synth_data(
        >>>     predictors=column_specifications,
        >>>     n_samples=100,
        >>>     text_prompt="The patient",
        >>> )
    """
    sequence = text_prompt if text_prompt else None

    for predictor_spec in predictors:
        for col_name, col_props in predictor_spec.items():
            # np.nan objects turn into "nan" strings in the real life dataframe.
            # imitate this in the synthetic data as well.
            if "nan" in col_name:
                df = df.rename({col_name: col_name.replace("np.nan", "nan")}, axis=1)
                col_name = col_name.replace("np.nan", "nan")  # noqa: PLW2901

            column_type = col_props["column_type"]

            df[col_name] = generate_col_from_specs(
                column_type=column_type, n_samples=n_samples, sequence=sequence, col_specs=col_props
            )

            # If column has min and/or max, floor and ceil appropriately
            if df[col_name].dtype not in ["datetime64[ns]"]:
                if "min" in col_props:
                    df[col_name] = df[col_name].clip(lower=col_props["min"])
                if "max" in col_props:
                    df[col_name] = df[col_name].clip(upper=col_props["max"])

    return df


if __name__ == "__main__":
    # Get project root directory
    column_specs = [
        {
            "dw_ek_borger": {"column_type": "id"},
            "raw_predictor": {"column_type": "uniform_float", "min": 0, "max": 10},
        }
    ]

    df = generate_data_columns(predictors=column_specs, n_samples=10_000)

    df.to_csv(PSYCOP_PKG_ROOT / "tests" / "test_data" / "synth_raw.csv", index=False)

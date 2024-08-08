import pandas as pd
import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    ConfusionMatrix,
    get_confusion_matrix_cells_from_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)


@shared_cache.cache()
def confusion_matrix_model(
    eval_df: pl.DataFrame, desired_positive_rate: float = 0.05
) -> ConfusionMatrix:
    eval_ds = eval_df

    df = eval_ds.rename({"y": "true", "y_hat_prob": "pred"}).to_pandas()

    df = pd.DataFrame(
        {
            "true": df["true"],
            "pred": get_predictions_for_positive_rate(
                desired_positive_rate=desired_positive_rate,
                y_hat_probs=df["pred"],  # type: ignore
            )[0],
        }
    )
    confusion_matrix = get_confusion_matrix_cells_from_df(df=df)
    return confusion_matrix

from collections.abc import Sequence
from typing import NewType

import pandas as pd
import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_sensitivity_by_timedelta_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)

SensitivityByTTEDF = NewType("SensitivityByTTEDF", pl.DataFrame)


@shared_cache().cache()
def sensitivity_by_time_to_event_model(
    eval_df: pl.DataFrame, pprs: Sequence[float] = (0.01, 0.03, 0.05)
) -> SensitivityByTTEDF:
    eval_dataset = eval_df.to_pandas()

    dfs = []
    for ppr in pprs:
        df = get_sensitivity_by_timedelta_df(
            y=eval_dataset.y,  # type: ignore
            y_hat=get_predictions_for_positive_rate(
                desired_positive_rate=ppr, y_hat_probs=eval_dataset.y_hat_prob
            )[0],
            time_one=eval_dataset["timestamp"],
            time_two=eval_dataset["timestamp_outcome"],
            direction="t2-t1",
            bins=range(0, 120, 12),
            bin_unit="D",
            bin_continuous_input=True,
        )

        # Convert to string to allow distinct scales for color
        df["actual_positive_rate"] = str(ppr)
        dfs.append(df)

    plot_df = pd.concat(dfs)
    return SensitivityByTTEDF(pl.from_pandas(plot_df))

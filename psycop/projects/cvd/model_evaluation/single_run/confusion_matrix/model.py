from dataclasses import dataclass

import pandas as pd

from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    ConfusionMatrix,
    get_confusion_matrix_cells_from_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import (
    RunSelector,
    SingleRunModel,
)


@dataclass(frozen=True)
class ConfusionMatrixModel(SingleRunModel):
    desired_positive_rate: float = 0.05

    def __call__(self, run: RunSelector) -> ConfusionMatrix:
        eval_ds = self._get_eval_df(run)

        df = eval_ds.rename({"y": "true", "y_hat_prob": "pred"}).to_pandas()

        df = pd.DataFrame(
            {
                "true": eval_ds["y"],
                "pred": get_predictions_for_positive_rate(
                    desired_positive_rate=self.desired_positive_rate,
                    y_hat_probs=eval_ds["y_hat_prob"],  # type: ignore
                )[0],
            }
        )
        confusion_matrix = get_confusion_matrix_cells_from_df(df=df)
        return confusion_matrix

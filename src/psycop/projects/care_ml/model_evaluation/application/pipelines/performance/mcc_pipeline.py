from pathlib import Path

import pandas as pd
from care_ml.model_evaluation.config import (
    EVAL_RUN,
    TABLES_PATH,
    TEXT_EVAL_RUN,
    TEXT_TABLES_PATH,
)
from care_ml.utils.best_runs import Run
from psycop.common.model_evaluation.binary.bootstrap_estimates import (
    bootstrap_estimates,
)
from sklearn.metrics import matthews_corrcoef


def bootstrap_mcc(
    y: pd.Series,
    y_hat: pd.Series,
    confidence_interval: bool = True,
    n_bootstraps: int = 1000,
) -> pd.Series:
    mcc = {"mcc": matthews_corrcoef(y, y_hat)}

    if confidence_interval:
        ci = bootstrap_estimates(
            matthews_corrcoef,
            n_bootstraps=n_bootstraps,
            ci_width=0.95,
            input_1=y,
            input_2=y_hat,
        )
        mcc["ci_lower"] = ci[0][0]
        mcc["ci_upper"] = ci[0][1]

    return pd.Series(mcc)


def mcc_pipeline(run: Run, path: Path):
    eval_ds = run.get_eval_dataset()

    mcc = bootstrap_mcc(
        eval_ds.y,
        eval_ds.get_predictions_for_positive_rate(
            desired_positive_rate=run.pos_rate,
        )[0],
    )

    mcc.to_csv(path / "mcc.csv")


if __name__ == "__main__":
    mcc_pipeline(EVAL_RUN, TABLES_PATH)
    mcc_pipeline(TEXT_EVAL_RUN, TEXT_TABLES_PATH)

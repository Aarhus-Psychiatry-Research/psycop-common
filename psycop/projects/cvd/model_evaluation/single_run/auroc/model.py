from dataclasses import dataclass

import numpy as np
import pandas as pd

from psycop.common.global_utils.cache import mem
from psycop.common.model_evaluation.binary.global_performance.roc_auc import bootstrap_roc
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import (
    RunSelector,
    SingleRunModel,
    get_eval_df,
)


@dataclass(frozen=True)
class AUROC:
    mean: float
    ci: tuple[float, float]
    fpr: np.ndarray  # type: ignore
    mean_tprs: np.ndarray  # type: ignore
    tprs_lower: np.ndarray  # type: ignore
    tprs_upper: np.ndarray  # type: ignore

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fpr": self.fpr,
                "tpr": self.mean_tprs,
                "tpr_lower": self.tprs_lower,
                "tpr_upper": self.tprs_upper,
            }
        )


@mem.cache()
def auroc_model(run: RunSelector, n_bootstraps: int = 5) -> AUROC:
    eval_df = get_eval_df(run)
    eval_dataset = eval_df.to_pandas()
    y = eval_dataset["y"]
    y_hat_probs = eval_dataset["y_hat_prob"]

    # We need a custom bootstrap implementation, because using scipy.bootstrap
    # on the roc_curve method will yield different fpr values for each resample,
    # and thus the tpr values will be interpolated on different fpr values. This
    # will result in arrays of different dimensions, which will cause an error.

    # Initialize lists for bootstrapped TPRs and FPRs
    tprs_bootstrapped, aucs_bootstrapped, base_fpr = bootstrap_roc(
        n_bootstraps=n_bootstraps, y=y, y_hat_probs=y_hat_probs
    )

    mean_tprs = tprs_bootstrapped.mean(axis=0)
    se_tprs = tprs_bootstrapped.std(axis=0) / np.sqrt(n_bootstraps)

    # Calculate confidence interval for TPR over all FPRs
    tprs_upper = mean_tprs + se_tprs
    tprs_lower = mean_tprs - se_tprs

    # Calculate confidence interval for AUC
    auc_mean = np.mean(aucs_bootstrapped)
    auc_se = np.std(aucs_bootstrapped) / np.sqrt(n_bootstraps)
    auc_ci = [auc_mean - 1.96 * auc_se, auc_mean + 1.96 * auc_se]

    return AUROC(
        mean=auc_mean,  # type: ignore
        ci=(auc_ci[0], auc_ci[1]),
        fpr=base_fpr,
        mean_tprs=mean_tprs,
        tprs_lower=tprs_lower,
        tprs_upper=tprs_upper,
    )

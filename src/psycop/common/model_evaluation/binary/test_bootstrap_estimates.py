from psycop.common.model_evaluation.binary.bootstrap_estimates import (
    bootstrap_estimates,
)
from psycop.common.test_utils.str_to_df import str_to_df
from sklearn.metrics import recall_score, roc_auc_score


def test_boostrap_estimates():
    input_df = str_to_df(
        """id,y,y_hat,y_pred,
        1,1,1.0,1,
        1,1,1.0,1,
        1,0,0.5,0,
        1,0,0.5,0,
        1,0,0.5,0,""",
    )

    auroc_df_with_ci = bootstrap_estimates(
        roc_auc_score,
        n_bootstraps=1,
        ci_width=0.95,
        input_1=input_df["y"],
        input_2=input_df["y_hat"],
    )

    assert auroc_df_with_ci["ci"] == (1.0, 1.0)

    sensitivity_df_with_ci = bootstrap_estimates(
        recall_score,
        n_bootstraps=5,
        ci_width=0.95,
        input_1=input_df["y"],
        input_2=input_df["y_pred"],
    )

    assert sensitivity_df_with_ci["ci"] == (1.0, 1.0)

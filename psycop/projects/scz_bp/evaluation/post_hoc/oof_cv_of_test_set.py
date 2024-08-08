import polars as pl
from sklearn.metrics import roc_auc_score

from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_best_run_in_experiment,
)

if __name__ == "__main__":
    test_set_experiment_name = "sczbp/test_tfidf_1000"
    cv_experiment_name = "sczbp/test_tfidf_1000_cv"

    test_set_df = scz_bp_get_eval_ds_from_best_run_in_experiment(
        experiment_name=test_set_experiment_name, model_type="joint"
    ).to_polars()
    cv_df = scz_bp_get_eval_ds_from_best_run_in_experiment(
        experiment_name=cv_experiment_name, model_type="joint"
    ).to_polars()

    only_midt_oof = cv_df.filter(pl.col("pred_time_uuids").is_in(test_set_df["pred_time_uuids"]))

    roc_auc_score(only_midt_oof["y"], y_score=only_midt_oof["y_hat_probs"])

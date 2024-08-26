


from sklearn.metrics import roc_auc_score
from psycop.common.model_evaluation.binary.bootstrap_estimates import bootstrap_estimates
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import scz_bp_get_eval_ds_from_best_run_in_experiment


d = {"joint training with synth" : "sczbp/structured_text_xgboost_ddpm",
 "joint training text only" : "sczbp/tfidf_1000-xgboost",
 "joint training structured only" : "sczbp/structured_only-xgboost",
 "joint random split": "sczbp/test_random_split"}


for name, experiment_name in d.items():
    eval_df =  scz_bp_get_eval_ds_from_best_run_in_experiment(
            experiment_name=experiment_name, model_type="joint"
        )
    auc_median = roc_auc_score(y_true=eval_df.y.to_numpy(), y_score=eval_df.y_hat_probs.to_numpy())
    ci = bootstrap_estimates(metric=roc_auc_score, input_1=eval_df.y, input_2=eval_df.y_hat_probs) #type: ignore

    print(f"{name}\n\tAUC: {auc_median:.2f}, ci low {ci[0][0]:.2f}, ci high {ci[0][1]:.2f}")
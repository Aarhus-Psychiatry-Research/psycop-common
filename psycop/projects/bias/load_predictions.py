from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame, MlflowClientWrapper

def load_ect_predictions():
    return MlflowClientWrapper().get_best_run_from_experiment(
                experiment_name="ECT hparam, structured_only, xgboost, no lookbehind filter",
                metric="all_oof_BinaryAUROC",
            ).eval_frame().frame


def load_scz_predictions():
    return MlflowClientWrapper().get_best_run_from_experiment(
                experiment_name="sczbp/test_scz_structured_text",
                metric="all_oof_BinaryAUROC",
            ).eval_frame().frame

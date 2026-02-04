from wasabi import Printer

from psycop.common.model_training.application_modules.train_model.main import train_model

msg = Printer(timestamp=True)

if __name__ == "__main__":
    from copy import copy

    from psycop.projects.forced_admission_outpatient.old_project_code.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    run = copy(get_best_eval_pipeline())
    run.name = "xgboost_full_dataset"

    msg.divider("Training with best params")
    cfg = run.inputs.cfg
    cfg.model.model_config["frozen"] = False
    cfg.data.splits_for_training = ["train", "val"]
    cfg.data.splits_for_evaluation = None
    cfg.n_crossval_splits = 10
    best_auc = train_model(cfg=cfg)
    print(f"Best AUC: {best_auc}")

    msg.divider("Training with default xgboost params")
    cfg.preprocessing.post_split.imputation_method = None
    cfg.preprocessing.post_split.scaling = None
    cfg.preprocessing.post_split.feature_selection.name = None
    cfg.model.args = {
        "n_estimators": 100,
        "alpha": 0,
        "lambda": 1,
        "max_depth": 30,
        "learning_rate": 0.3,
        "gamma": 0,
        "grow_policy": "depthwise",
    }
    default_auroc_2 = train_model(cfg=cfg)

    msg.divider("Training with default xgboost params (max depth = 5)")
    cfg.model.args = {"max_depth": 5}
    default_auroc_5 = train_model(cfg=cfg)

    msg.divider("Training with default xgboost params (max depth = 10)")
    cfg.model.args = {"max_depth": 10}
    default_auroc_10 = train_model(cfg=cfg)

    msg.divider("Training with default xgboost params (max depth = 20)")
    cfg.model.args = {"max_depth": 20}
    default_auroc_20 = train_model(cfg=cfg)

    msg.divider("Training with default xgboost params (max depth = 30)")
    cfg.model.args = {"max_depth": 30}
    default_auroc_30 = train_model(cfg=cfg)

    print(f"Best AUC: {best_auc}")
    print(f"Default AUC (max depth = 2): {default_auroc_2}")
    print(f"Default AUC (max depth = 5): {default_auroc_5}")
    print(f"Default AUC (max depth = 10): {default_auroc_10}")
    print(f"Default AUC (max depth = 20): {default_auroc_20}")
    print(f"Default AUC (max depth = 30): {default_auroc_30}")

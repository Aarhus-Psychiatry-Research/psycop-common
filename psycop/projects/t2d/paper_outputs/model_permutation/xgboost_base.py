from wasabi import Printer

from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)

msg = Printer(timestamp=True)

if __name__ == "__main__":
    from copy import copy

    from psycop.projects.t2d.paper_outputs.selected_runs import (
        get_best_eval_pipeline,
    )

    run = copy(get_best_eval_pipeline())
    run.name = "xgboost_full_dataset"

    msg.divider("Training with best params")
    cfg = run.inputs.cfg
    best_auc = train_model(cfg=cfg)
    print(f"Best AUC: {best_auc}")

    msg.divider("Training with default xgboost params")
    cfg.model.Config.allow_mutation = True
    cfg.model.args = {
        "n_estimators": 100,
        "alpha": 0,
        "lambda": 1,
        "max_depth": 6,
        "learning_rate": 0.3,
        "gamma": 0,
        "grow_policy": "depthwise",
    }
    default_auroc = train_model(cfg=cfg)
    print(f"Default AUC: {default_auroc}")

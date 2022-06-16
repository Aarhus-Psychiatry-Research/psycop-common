from pathlib import Path

import hydra
import src.features.post_process as post_process
import wandb
from sklearn.metrics import roc_auc_score
from src.features.load_features import *
from src.utils import generate_predictions, impute, log_tpr_by_time_to_event
from wasabi import Printer


@hydra.main(
    version_base=None, config_path=Path(".") / "conf", config_name="train_config"
)
def main(cfg):
    if cfg.evaluation.wandb:
        run = cfg.evaluation.wandb.init(project="psycop-t2d", reinit=True)
        run.config.update(cfg)

    OUTCOME_COL_NAME = f"t2d_within_{cfg.training.lookahead_days}_days_max_fallback_0"
    PREDICTED_OUTCOME_COL_NAME = f"pred_{OUTCOME_COL_NAME}"

    if cfg.post_processing.load_all:
        n_to_load = None
    else:
        n_to_load = 5_000

    outcome_timestamp_col_name = f"timestamp_{OUTCOME_COL_NAME}"

    cols_to_drop_before_training = [
        "dw_ek_borger",
        "prediction_time_uuid",
        outcome_timestamp_col_name,
    ]

    # Val set
    val_combined = load_val(outcome_col_name=OUTCOME_COL_NAME, n_to_load=n_to_load)
    X_val, y_val = post_process.val(
        train_combined=val_combined,
        outcome_col_name=OUTCOME_COL_NAME,
        min_lookahead_days=cfg.post_processing.min_lookahead_days,
        min_lookbehind_days=cfg.post_processing.min_lookbehind_days,
        cols_to_drop_before_training=cols_to_drop_before_training,
        convert_all_cols_to_binary=cfg.post_processing.force_all_binary,
    )
    eval_X = X_val.copy()

    # Train set
    train_combined = load_train(outcome_col_name=OUTCOME_COL_NAME, n_to_load=n_to_load)
    X_train, y_train = post_process.train(
        df_combined=train_combined,
        outcome_col_name=OUTCOME_COL_NAME,
        min_lookahead_days=cfg.post_processing.min_lookahead_days,
        min_lookbehind_days=cfg.post_processing.min_lookbehind_days,
        cols_to_drop_before_training=cols_to_drop_before_training,
        convert_all_cols_to_binary=cfg.post_processing.force_all_binary,
    )

    # Make a copy of the datasets so you can inspect during debugging
    if cfg.post_processing.impute_all:
        train_X_processed, val_X_processed = impute(train_X=X_train, val_X=X_val)
    else:
        train_X_processed, val_X_processed = X_train, X_val

    y_preds, y_probas, model = generate_predictions(
        y_train, train_X_processed, val_X_processed
    )

    # Evaluation

    if cfg.evaluation.wandb:
        run.log({"roc_auc_unweighted": round(roc_auc_score(y_val, y_probas[:, 1]), 3)})

        wandb.sklearn.plot_classifier(
            model,
            X_train=X_train,
            X_test=X_val,
            y_train=y_train,
            y_test=y_val,
            y_pred=y_preds,
            y_probas=y_probas,
            labels=[0, 1],
            model_name="XGB",
            feature_names=train_X_processed.columns,
        )

    eval_df = eval_X
    eval_df[OUTCOME_COL_NAME] = y_val
    eval_df[PREDICTED_OUTCOME_COL_NAME] = y_preds

    log_tpr_by_time_to_event(
        df=eval_df,
        outcome_col_name=OUTCOME_COL_NAME,
        predicted_outcome_col_name=PREDICTED_OUTCOME_COL_NAME,
        outcome_timestamp_col_name=outcome_timestamp_col_name,
        prediction_timestamp_col_name="timestamp",
        bins=[0, 1, 7, 14, 28, 182, 365, 730, 1825],
    )

    if cfg.evaluation.wandb:
        run.finish()


if __name__ == "__main__":
    msg = Printer(timestamp=True)

    main()

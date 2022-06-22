from pathlib import Path

import hydra
from sklearn.metrics import roc_auc_score
from wasabi import Printer

import psycopt2d.features.post_process as post_process
import wandb
from psycopt2d.features.load_features import load_dataset
from psycopt2d.utils import (
    calculate_performance_metrics,
    flatten_nested_dict,
    generate_predictions,
    impute,
    log_tpr_by_time_to_event,
)


@hydra.main(
    version_base=None,
    config_path=Path(".") / "conf",
    config_name="train_config",
)
def main(cfg):
    if cfg.evaluation.wandb:
        run = wandb.init(project="psycop-t2d", reinit=True)

        # Flatten nested dict to support wandb filtering
        run.config.update(flatten_nested_dict(cfg, sep="."))

    OUTCOME_COL_NAME = f"t2d_within_{cfg.training.lookahead_days}_days_max_fallback_0"
    PREDICTED_OUTCOME_COL_NAME = f"pred_{OUTCOME_COL_NAME}"
    OUTCOME_TIMESTAMP_COL_NAME = f"timestamp_{OUTCOME_COL_NAME}"
    PREDICTED_PROBABILITY_COL_NAME = f"pred_prob_{OUTCOME_COL_NAME}"

    if cfg.post_processing.load_all:
        n_to_load = None
    else:
        n_to_load = 5_000

    cols_to_drop_before_training = cfg.training.cols_to_drop_before_training + [
        OUTCOME_TIMESTAMP_COL_NAME,
    ]

    # Val set
    X_val, y_val = load_dataset(
        split_name="val",
        n_to_load=n_to_load,
        outcome_col_name=OUTCOME_COL_NAME,
    )

    (X_val, y_val, X_val_eval, y_val_eval) = post_process.combined(
        X=X_val,
        y=y_val,
        outcome_col_name=OUTCOME_COL_NAME,
        min_lookahead_days=cfg.post_processing.val.min_lookahead_days,
        min_lookbehind_days=cfg.post_processing.val.min_lookbehind_days,
        convert_all_cols_to_binary=cfg.post_processing.force_all_binary,
        cols_to_drop=cols_to_drop_before_training,
        drop_if_any_diabetes_before_date=cfg.post_processing.val.drop_if_any_diabetes_before_date,
    )

    # Train set
    X_train, y_train = load_dataset(
        split_name="train",
        outcome_col_name=OUTCOME_COL_NAME,
        n_to_load=n_to_load,
    )

    (X_train, y_train, X_train_eval, y_train_eval) = post_process.combined(
        X=X_train,
        y=y_train,
        outcome_col_name=OUTCOME_COL_NAME,
        min_lookahead_days=cfg.post_processing.train.min_lookahead_days,
        min_lookbehind_days=cfg.post_processing.train.min_lookbehind_days,
        convert_all_cols_to_binary=cfg.post_processing.force_all_binary,
        cols_to_drop=cols_to_drop_before_training,
        drop_if_any_diabetes_before_date=cfg.post_processing.val.drop_if_any_diabetes_before_date,
    )

    # Consider moving impute to post_process
    if cfg.post_processing.impute_all:
        train_X_imputed, val_X_imputed = impute(train_X=X_train, val_X=X_val)
    else:
        train_X_imputed, val_X_imputed = X_train, X_val

    y_preds, y_probas, model = generate_predictions(
        y_train,
        train_X_imputed,
        val_X_imputed,
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
            feature_names=X_train.columns,
        )

    eval_df = X_val_eval
    eval_df[OUTCOME_COL_NAME] = y_val_eval
    eval_df[PREDICTED_OUTCOME_COL_NAME] = y_preds
    eval_df[PREDICTED_PROBABILITY_COL_NAME] = y_probas

    if cfg.evaluation.wandb:
        log_tpr_by_time_to_event(
            eval_df_combined=eval_df,
            outcome_col_name=OUTCOME_COL_NAME,
            predicted_outcome_col_name=PREDICTED_OUTCOME_COL_NAME,
            outcome_timestamp_col_name=OUTCOME_TIMESTAMP_COL_NAME,
            prediction_timestamp_col_name="timestamp",
            bins=[0, 1, 7, 14, 28, 182, 365, 730, 1825],
        )
        performance_metrics = calculate_performance_metrics(
            eval_df,
            outcome_col_name=OUTCOME_COL_NAME,
            prediction_probabilities_col_name=PREDICTED_PROBABILITY_COL_NAME,
            id_col_name="dw_ek_borger",
        )

        run.log(performance_metrics)
        run.finish()


if __name__ == "__main__":
    msg = Printer(timestamp=True)

    main()

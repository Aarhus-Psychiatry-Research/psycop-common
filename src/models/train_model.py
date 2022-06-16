import datetime as dt
from sklearn.metrics import roc_auc_score
from src.features.load_features import *
from wasabi import Printer
import wandb

from src.utils import (
    convert_all_to_binary,
    drop_records_if_datediff_days_smaller_than,
    difference_in_days,
    generate_predictions,
    impute,
    log_tpr_by_time_to_event,
)


if __name__ == "__main__":
    msg = Printer(timestamp=True)

    LOG = False

    if LOG:
        run = wandb.init(project="psycop-t2d", reinit=True)

    PARAMS = {
        "impute_all": False,
        "load_all": False,
        "force_all_binary": False,
        "lookahead_days": 1826.25,
        "lookbehind_days": 9999,
        "min_lookahead_days": 5,
        "min_lookbehind_days": 5,
        "washin_days": 0,
    }

    if LOG:
        run.config.update(PARAMS)

    OUTCOME_COL_NAME = f"t2d_within_{PARAMS['lookahead_days']}_days_max_fallback_0"
    PREDICTED_OUTCOME_COL_NAME = f"pred_{OUTCOME_COL_NAME}"

    if PARAMS["load_all"]:
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
    X_val, y_val = load_val(outcome_col_name=OUTCOME_COL_NAME, n_to_load=n_to_load)
    eval_X = X_val.copy()

    # Train set
    X_train, y_train = load_train(
        outcome_col_name=OUTCOME_COL_NAME, n_to_load=n_to_load
    )

    # Prep for training
    for ds in X_train, X_val:
        # Handle minimum lookahead
        if PARAMS["min_lookahead_days"] is not False:
            drop_records_if_datediff_days_smaller_than(
                df=ds,
                t2_col_name=f"timestamp_t2d_within_{PARAMS['lookahead_days']}_days_max_fallback_0",
                t1_col_name="timestamp",
                threshold_days=PARAMS["lookahead_days"],
                inplace=True,
            )

        # Handle minimum lookahead
        if PARAMS["min_lookbehind_days"] is not False:
            _first_pred_time_col_name = "timestamp_first_prediction_time"

            ds[_first_pred_time_col_name] = ds["timestamp"].min()

            ds["difference_in_days"] = difference_in_days(
                ds["timestamp"], ds[_first_pred_time_col_name]
            )

            drop_records_if_datediff_days_smaller_than(
                df=ds,
                t2_col_name=f"timestamp",
                t1_col_name=_first_pred_time_col_name,
                threshold_days=PARAMS["min_lookahead_days"],
                inplace=True,
            )

            ds.drop([_first_pred_time_col_name], inplace=True, axis=1)

        # Drop columns that won't generalize
        msg.info("Dropping columns that won't generalise")
        ds.drop(cols_to_drop_before_training, axis=1, errors="ignore", inplace=True)

        # Handle making binary
        if PARAMS["force_all_binary"]:
            convert_all_to_binary(ds, skip=["age_in_years", "male"])

        timestamp_cols = [col for col in ds.columns if "timestamp" in col]
        msg.info("Converting timestamps to ordinal")
        for colname in timestamp_cols:
            ds[colname] = ds[colname].map(dt.datetime.toordinal)

    # Make a copy of the datasets so you can inspect during debugging
    if PARAMS["impute_all"]:
        train_X_processed, val_X_processed = impute(train_X=X_train, val_X=X_val)
    else:
        train_X_processed, val_X_processed = X_train, X_val

    y_preds, y_probas, model = generate_predictions(
        y_train, train_X_processed, val_X_processed
    )

    # Evaluation
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

    if LOG:
        run.log({"roc_auc_unweighted": round(roc_auc_score(y_val, y_probas[:, 1]), 3)})

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

    if LOG:
        run.finish()

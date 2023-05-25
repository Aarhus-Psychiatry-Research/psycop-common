import polars as pl
from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.projects.t2d.paper_outputs.config import BEST_EVAL_PIPELINE, ESTIMATES_PATH

if __name__ == "__main__":
    cfg: FullConfigSchema = BEST_EVAL_PIPELINE.cfg

    # Create the dataset with only HbA1c-predictors
    df: pl.LazyFrame = pl.concat(
        BEST_EVAL_PIPELINE.get_flattened_split_as_lazyframe(split) for split in ["train", "val"]  # type: ignore
    )

    pred_cols = [c for c in df.columns if c.startswith(cfg.data.pred_prefix)]
    pred_cols_sans_sex = [c for c in pred_cols if "sex" not in c]

    boolean_df = df
    for col in pred_cols_sans_sex:
        boolean_df = df.with_columns(
            pl.when(pl.col(col).is_not_null()).then(1).otherwise(0).alias(col),
        )

    boolean_df = boolean_df.collect()

    path_prefix = "boolean"

    boolean_pred_dir = BEST_EVAL_PIPELINE.eval_dir / f"{path_prefix}_preds"
    boolean_pred_dir.mkdir(parents=True, exist_ok=True)
    boolean_pred_path = boolean_pred_dir / f"{path_prefix}_preds.parquet"
    boolean_df.write_parquet(boolean_pred_path)

    # Point the model at that dataset
    cfg.data.Config.allow_mutation = True
    cfg.data.dir = str(boolean_pred_dir)
    cfg.data.splits_for_training = [f"{path_prefix}"]
    roc_auc = train_model(cfg=cfg)

    ESTIMATES_PATH.mkdir(parents=True, exist_ok=True)

    # Write AUROC
    with (ESTIMATES_PATH / f"{path_prefix}.txt").open("a") as f:
        f.write(str(roc_auc))
        f.write(str(boolean_df.columns))

import polars as pl
from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.projects.t2d.paper_outputs.selected_runs import (
    BEST_EVAL_PIPELINE,
)
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun


def evaluate_pipeline_with_boolean_features(run: PipelineRun):
    cfg: FullConfigSchema = run.inputs.cfg

    # Create the dataset with only HbA1c-predictors
    df: pl.LazyFrame = pl.concat(
        run.get_flattened_split_as_lazyframe(split) for split in ["train", "val"]  # type: ignore
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

    boolean_pred_dir = run.paper_outputs.paths.estimates / f"{path_prefix}_preds"
    boolean_pred_dir.mkdir(parents=True, exist_ok=True)
    boolean_pred_path = boolean_pred_dir / f"{path_prefix}_preds.parquet"
    boolean_df.write_parquet(boolean_pred_path)

    # Point the model at that dataset
    cfg.data.Config.allow_mutation = True
    cfg.data.dir = str(boolean_pred_dir)
    cfg.data.splits_for_training = [f"{path_prefix}"]
    roc_auc = train_model(cfg=cfg)

    # Write AUROC
    with (run.paper_outputs.paths.estimates / f"{path_prefix}.txt").open("a") as f:
        f.write(str(roc_auc))
        f.write(str(boolean_df.columns))


if __name__ == "__main__":
    evaluate_pipeline_with_boolean_features(run=BEST_EVAL_PIPELINE)

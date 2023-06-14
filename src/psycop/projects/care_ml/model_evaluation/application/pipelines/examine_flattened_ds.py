from care_ml.model_evaluation.best_runs import best_run
from care_ml.model_evaluation.data.load_true_data import load_fullconfig
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.data_loader.data_loader import DataLoader

if __name__ == "__main__":
    cfg: FullConfigSchema = load_fullconfig(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
    )
    flattened_ds = DataLoader(data_cfg=cfg.data).load_dataset_from_dir(
        split_names="val",
    )

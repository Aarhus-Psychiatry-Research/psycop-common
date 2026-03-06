from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.t2d_extended.model_training.populate_t2d_registry import populate_with_t2d_extended_registry
import polars as pl



if __name__ == "__main__":
    
    training_start_date="2013-01-01"
    training_end_date="2018-01-01"
    evaluation_interval=(f"2018-01-01", f"2018-12-31")

    populate_baseline_registry()
    populate_with_t2d_extended_registry

    run_path = f"E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/{training_start_date}_{training_end_date}_{evaluation_interval[0]}_{evaluation_interval[1]}"

    cfg = PsycopConfig().from_disk(f"{run_path}/config.cfg")

    dataset_dirs = cfg.retrieve("trainer.training_data.paths")

    feature_set_df = pl.read_parquet(dataset_dirs[0])

    print("hey")

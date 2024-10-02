from pathlib import Path

import polars as pl

from psycop.common.model_training_v2.config.config_utils import PsycopConfig

if __name__ == "__main__":
    # read parquet file
    cfg = PsycopConfig().from_disk(
        Path(__file__).parent.parent / "model_training" / "t2d_bigdata.cfg"
    )
    dataset_dirs = cfg.retrieve("trainer.training_data.paths")

    df = pl.read_parquet(dataset_dirs[0])

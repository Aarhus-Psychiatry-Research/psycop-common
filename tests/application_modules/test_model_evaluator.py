from pathlib import Path

import pandas as pd
from psycop_model_training.application_modules.train_model.main import train_model
from psycop_model_training.config_schemas.full_config import FullConfigSchema


def test_saving_results_to_parquet(
    muteable_test_config: FullConfigSchema,
    tmp_path: Path,
):
    """Test that model performance is saved to a parquet file for querying."""
    cfg = muteable_test_config

    for _ in [0, 1]:
        # Run twice to ensure that we can also append to a file
        train_model(cfg, override_output_dir=tmp_path / "run_eval")

    run_performance_path = list(tmp_path.glob(r"*.parquet"))[0]
    run_performance_df = pd.read_parquet(run_performance_path)

    for info in ["run_name", "roc_auc", "timestamp", "lookahead_days", "model_name"]:
        assert info in run_performance_df.columns

    assert len(run_performance_df["run_name"].unique()) == 2

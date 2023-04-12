import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import dill as pkl
import pandas as pd
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.training_output.dataclasses import EvalDataset, PipeMetadata
from psycop_model_training.utils.utils import write_df_to_file
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


def dump_to_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Pickles an object to a file.

    Args:
        obj (Any): Object to pickle.
        path (str): Path to pickle file.
    """
    with Path(path).open(mode="wb") as f:
        pkl.dump(obj, f)


class ArtifactsToDiskSaver:
    """Class for saving artifacts to disk."""

    def __init__(self, dir_path: Path):
        self.dir_path = dir_path

        dir_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def eval_dataset_to_disk(eval_dataset: EvalDataset, file_path: Path) -> None:
        """Write EvalDataset to disk.

        Handles csv and parquet files based on suffix.
        """
        # Add base columns and custom columns
        df_template = {
            col_name: series
            for col_name, series in eval_dataset.__dict__.items()
            if series is not None
        }

        # Check if custom_columns attribute exists
        if (
            hasattr(eval_dataset, "custom_columns")
            and eval_dataset.custom_columns is not None
        ):
            df_template |= {
                col_name: series
                for col_name, series in eval_dataset.custom_columns.items()
                if series is not None
            }

        # Remove items that aren't series, e.g. the top level CustomColumns object
        template_filtered = {
            k: v for k, v in df_template.items() if isinstance(v, pd.Series)
        }

        df = pd.DataFrame(template_filtered)

        write_df_to_file(df=df, file_path=file_path)

    def save(
        self,
        cfg: Optional[FullConfigSchema],
        eval_dataset: Optional[EvalDataset],
        pipe_metadata: Optional[PipeMetadata],
        pipe: Optional[Pipeline],
    ) -> None:
        """Saves prediction dataframe, hydra config and feature names to
        disk."""
        if eval_dataset is not None:
            self.eval_dataset_to_disk(
                eval_dataset,
                self.dir_path / "evaluation_dataset.parquet",
            )

        if cfg is not None:
            dump_to_pickle(cfg, self.dir_path / "cfg.pkl")
            
            with (self.dir_path / "cfg.json").open(mode="w") as f:
                cfg_dict = cfg.json()
                json.dump(cfg_dict, f)

        if pipe_metadata is not None:
            dump_to_pickle(pipe_metadata, self.dir_path / "pipe_metadata.pkl")

        if pipe is not None:
            dump_to_pickle(pipe, self.dir_path / "pipe.pkl")

        log.info(  # pylint: disable=logging-fstring-interpolation
            f"Saved evaluation dataset, cfg and pipe metadata to {self.dir_path}",
        )

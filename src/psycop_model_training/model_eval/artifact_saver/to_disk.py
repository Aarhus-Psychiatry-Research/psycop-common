from typing import Any, Union, Optional
import dill as pkl

import pandas as pd

from psycop_model_training.model_eval.dataclasses import EvalDataset, PipeMetadata
from psycop_model_training.utils.config_schemas.full_config import FullConfigSchema
from psycop_model_training.utils.utils import write_df_to_file

import logging
from wandb import Run
from pathlib import Path

log = logging.getLogger(__name__)

class ArtifactsToDiskSaver:
    def __init__(self, run: Run, dir_path: Path):
        self.run = run
        self.dir_path = dir_path

        dir_path.mkdir(parents=True, exist_ok=True)

    def eval_dataset_to_disk(eval_dataset: EvalDataset, file_path: Path) -> None:
        """Write EvalDataset to disk.

        Handles csv and parquet files based on suffix.
        """
        # Add base columns and custom columns
        df_template = {
                          col_name: series
                          for col_name, series in eval_dataset.__dict__.items()
                          if series is not None
                      } | {
                          col_name: series
                          for col_name, series in eval_dataset.custom.__dict__.items()
                          if series is not None
                      }

        # Remove items that aren't series, e.g. the top level CustomColumns object
        template_filtered = {
            k: v for k, v in df_template.items() if isinstance(v, pd.Series)
        }

        df = pd.DataFrame(template_filtered)

        write_df_to_file(df=df, file_path=file_path)


    def save(self, cfg: Optional[FullConfigSchema], eval_dataset: Optional[EvalDataset], pipe_metadata: Optional[PipeMetadata], pipe: Optional[Pipeline]) -> None:
        """Saves prediction dataframe, hydra config and feature names to disk."""
        if eval_dataset is not None:
            self.eval_dataset_to_disk(eval_dataset, self.dir_path / "evaluation_dataset.parquet")

        if cfg is not None:
            dump_to_pickle(cfg, self.dir_path / "cfg.pkl")

        if pipe_metadata is not None:
            dump_to_pickle(pipe_metadata, self.dir_path / "pipe_metadata.pkl")

        if pipe is not None:
            dump_to_pickle(pipe, self.dir_path / "pipe.pkl")

        log.info(f"Saved evaluation dataset, cfg and pipe metadata to {self.dir_path}")

def dump_to_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Pickles an object to a file.

    Args:
        obj (Any): Object to pickle.
        path (str): Path to pickle file.
    """
    with open(path, "wb") as f:
        pkl.dump(obj, f)




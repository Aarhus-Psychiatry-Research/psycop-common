import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.model_training.config_schemas.conf_utils import FullConfigSchema
from psycop.common.model_training.training_output.dataclasses import EvalDataset

EVAL_ROOT = Path(__file__).parent.parent / "outputs_for_publishing"


def load_file_from_pkl(file_path: Path) -> Any:
    with file_path.open("rb") as f:
        return pickle.load(f)


def df_to_eval_dataset(
    df: pd.DataFrame,
    custom_columns: Optional[Sequence[str]],
) -> EvalDataset:
    """Convert dataframe to EvalDataset."""
    return EvalDataset(
        ids=df["ids"],
        y=df["y"],
        y_hat_probs=df["y_hat_probs"],
        pred_timestamps=df["pred_timestamps"],
        outcome_timestamps=df["outcome_timestamps"],
        age=df["age"],
        is_female=df["is_female"],
        exclusion_timestamps=df["exclusion_timestamps"],
        pred_time_uuids=df["pred_time_uuids"],
        custom_columns={col: df[col] for col in custom_columns}
        if custom_columns
        else None,
    )


@dataclass
class RunGroup:
    name: str

    @property
    def group_dir(self) -> Path:
        return Path(f"E:/shared_resources/t2d/model_eval/{self.name}")

    @property
    def flattened_ds_dir(self) -> Path:
        oldest_run = min(self.group_dir.iterdir(), key=lambda f: f.stat().st_mtime)

        config_path = oldest_run / "cfg.json"

        with config_path.open() as f:
            config_str = json.load(f)
            config_dict = json.loads(config_str)

        return Path(config_dict["data"]["dir"])

    @property
    def all_runs_performance_df(self) -> pd.DataFrame:
        run_performance_files = self.group_dir.glob("*.parquet")

        concatenated_performance_df = pd.concat(
            pd.read_parquet(parquet_file) for parquet_file in run_performance_files
        )

        return concatenated_performance_df

    def get_best_runs_by_lookahead(self) -> pl.DataFrame:
        df = pl.from_pandas(self.all_runs_performance_df)

        return (
            df.groupby(["lookahead_days", "model_name"])
            .agg(pl.all().sort_by("roc_auc", descending=True).first())
            .sort(["model_name", "lookahead_days"])
        )


SplitNames = Literal["train", "test", "val"]


@dataclass
class PipelineInputs:
    group: RunGroup
    eval_dir: Path
    additional_cfg_keys: dict[str, dict[str, Any]] | None

    def get_cfg_as_json(self) -> dict[str, Any]:
        # Load json
        path = self.eval_dir / "cfg.json"
        source_json = json.loads(json.loads(path.read_text()))

        if self.additional_cfg_keys:
            for k in self.additional_cfg_keys:
                source_v = source_json[k]
                additional_v = self.additional_cfg_keys[k]
                result_v = source_v.update(additional_v)
                source_json.update(result_v)

        return source_json

    def _get_flattened_split_path(self, split: SplitNames) -> Path:
        matches = list(self.group.flattened_ds_dir.glob(f"*{split}*.parquet"))

        if len(matches) > 1:
            raise ValueError("More than one matching split file found")
        return matches[0]

    def get_flattened_split_as_pd(self, split: SplitNames) -> pd.DataFrame:
        return pd.read_parquet(self._get_flattened_split_path(split=split))

    def get_flattened_split_as_lazyframe(self, split: SplitNames) -> pl.LazyFrame:
        return pl.scan_parquet(self._get_flattened_split_path(split=split))

    @property
    def cfg(self) -> FullConfigSchema:
        """
        Loads the config object for the given pipeline.
        Loading the json instead of the .pkl makes us independent
            of whether the imports in psycop-common model-training have changed

        Args:
            extension_dict: Add this dict to the cfg object. Helpful when the cfg object has changed, e.g. a parameter has been added, and you want to add that parameter without changing the full cfg.

        TODO: Note that this means assigning to the cfg property does nothing, since it's recomputed every time it's called
        """
        pipeline_dict = self.get_cfg_as_json()

        if "project_path" not in pipeline_dict["project"].keys():
            pipeline_dict["project"]["project_path"] = OVARTACI_SHARED_DIR / "t2d"

        return FullConfigSchema.parse_obj(pipeline_dict)


@dataclass
class PipelineOutputs:
    name: str
    group: RunGroup
    dir_path: Path

    def get_eval_dataset(
        self,
        custom_columns: Optional[Sequence[str]] = None,
    ) -> EvalDataset:
        df = pd.read_parquet(self.dir_path / "evaluation_dataset.parquet")

        eval_dataset = df_to_eval_dataset(df, custom_columns=custom_columns)

        return eval_dataset

    def get_auroc(self) -> float:
        df = self.group.all_runs_performance_df
        self_run = df[df["run_name"] == self.name]
        return self_run["roc_auc"].iloc[0]

    @property
    def pipe(self) -> Pipeline:
        return load_file_from_pkl(self.dir_path / "pipe.pkl")


@dataclass
class T2DArtifactNames:
    main_performance_figure: str = "t2d_main_performance_figure.png"
    main_robustness_figure: str = "t2d_main_robustness.png"
    performance_by_ppr: str = "t2d_performance_by_ppr.xlsx"


class PaperOutputPaths:
    def __init__(self, artifact_path: Path, create_output_paths_on_init: bool = True):
        self.artifact = artifact_path
        self.tables = self.artifact / "tables"
        self.figures = self.artifact / "figures"
        self.estimates = self.artifact / "estimates"

        if create_output_paths_on_init:
            for path in [self.artifact, self.tables, self.figures, self.estimates]:
                path.mkdir(parents=True, exist_ok=True)


class PaperOutputSettings:
    def __init__(
        self,
        name: str,
        pos_rate: float,
        model_type: str,
        lookahead_days: int,
        artifact_root: Optional[Path] = None,
        create_output_paths_on_init: bool = True,
    ):
        self.name = name
        self.pos_rate = pos_rate
        artifact_root = EVAL_ROOT if artifact_root is None else artifact_root
        self.artifact_path = (
            artifact_root / f"{lookahead_days}_{model_type}_{self.name}"
        )
        self.artifact_names = T2DArtifactNames()
        self.paths = PaperOutputPaths(
            self.artifact_path,
            create_output_paths_on_init=create_output_paths_on_init,
        )


class T2DPipelineRun:
    def __init__(
        self,
        name: str,
        group: RunGroup,
        pos_rate: float,
        additional_cfg_keys: dict[str, Any] | None = None,
        paper_outputs_path: Optional[Path] = None,
        create_output_paths_on_init: bool = True,
        additional_cfg_keys: dict[str, Any],
    ):
        self.name = name
        self.group = group
        pipeline_output_dir = self.group.group_dir / self.name

        self.inputs = PipelineInputs(
            group=group,
            eval_dir=pipeline_output_dir,
            additional_cfg_keys=additional_cfg_keys,
        )
        self.pipeline_outputs = PipelineOutputs(
            group=group,
            dir_path=pipeline_output_dir,
            name=self.name,
        )
        self.paper_outputs = PaperOutputSettings(
            name=name,
            pos_rate=pos_rate,
            artifact_root=paper_outputs_path,
            lookahead_days=self.inputs.cfg.preprocessing.pre_split.min_lookahead_days,
            model_type=self.model_type,
            create_output_paths_on_init=create_output_paths_on_init,
        )

    @property
    def model_type(self) -> str:
        return self.inputs.cfg.model.name

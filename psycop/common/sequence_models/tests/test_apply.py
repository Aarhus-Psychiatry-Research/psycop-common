from pathlib import Path

import polars as pl
import pytest
from torch import nn

from psycop.common.feature_generation.sequences.prediction_time_collater import (
    BasePredictionTimeCollater,
)
from psycop.common.sequence_models.aggregators import Aggregator
from psycop.common.sequence_models.apply import (
    apply,
)
from psycop.common.sequence_models.config_schema import (
    TrainerConfigSchema,
    TrainingConfigSchema,
)
from psycop.common.sequence_models.embedders.interface import PatientSliceEmbedder
from psycop.common.sequence_models.optimizers import LRSchedulerFn, OptimizerFn
from psycop.common.sequence_models.tasks.patientslice_classifier import (
    PatientSliceClassifier,
)
from psycop.common.sequence_models.tasks.patientslice_classifier_base import (
    BasePredictionTimeClassifier,
)

from .test_encoder_for_clf import skip_if_arm_within_docker
from .test_finetuning import FakePredictionTimeCollater


@pytest.fixture()
def dataset_collater() -> BasePredictionTimeCollater:
    return FakePredictionTimeCollater()


@pytest.fixture()
def patient_slice_classifier(
    embedder: PatientSliceEmbedder,
    encoder: nn.Module,
    aggregator: Aggregator,
    optimizer: OptimizerFn,
    lr_scheduler_fn: LRSchedulerFn,
) -> PatientSliceClassifier:
    return PatientSliceClassifier(
        embedder=embedder,
        encoder=encoder,
        aggregator=aggregator,
        num_classes=2,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler_fn,
    )


@pytest.fixture()
def training_config() -> TrainingConfigSchema:
    return TrainingConfigSchema(
        trainer=TrainerConfigSchema(
            accelerator="cpu",
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        ),
        batch_size=2,
        num_workers_for_dataloader=1,
    )


@skip_if_arm_within_docker
def test_apply(
    tmp_path: Path,
    patient_slice_classifier: BasePredictionTimeClassifier,
    dataset_collater: BasePredictionTimeCollater,
    training_config: TrainingConfigSchema,
):
    output_parquet_path = tmp_path / "output.parquet"
    apply(
        patient_slice_classifier,
        dataset_collater,
        training_config,
        output_parquet_path,
    )

    df = pl.read_parquet(output_parquet_path)

    assert df["pred_proba"].dtype == pl.Float32
    assert df["pred_time_uuid"].dtype == pl.Utf8
    uuid = df["pred_time_uuid"].to_list()
    assert len(uuid) == len(set(uuid))

import logging
from pathlib import Path

import lightning.pytorch as pl
import polars
from torch.utils.data import DataLoader

from psycop.common.feature_generation.sequences.patient_slice_collater import (
    BasePatientSliceCollater,
)
from psycop.common.feature_generation.sequences.prediction_time_collater import (
    BasePredictionTimeCollater,
)
from psycop.common.model_training import training
from psycop.common.sequence_models.config_schema import TrainingConfigSchema
from psycop.common.sequence_models.registry import Registry
from psycop.common.sequence_models.tasks.patientslice_classifier_base import (
    BasePatientSliceClassifier,
)

log = logging.getLogger(__name__)


def apply(
    model: BasePatientSliceClassifier,
    dataset_collater: BasePredictionTimeCollater,
    training_config: TrainingConfigSchema,
    output_parquet_path: Path,
) -> None:
    """
    Apply a model to a dataset and save the results to a dataframe.
    """
    # add predict_step to protocol
    # implement predict_step on model
    # make dataloader

    filter_fn = model.filter_and_reformat

    log.info("Preparing dataset")
    dataset = dataset_collater.get_dataset()
    dataset.filter_patients(filter_fn)
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=model.collate_fn,
        num_workers=training_config.num_workers_for_dataloader,
        persistent_workers=True,
    )

    trainer = pl.Trainer(**training_config.trainer.to_dict())
    predictions = trainer.predict(model, dataloader)

    df = polars.DataFrame(predictions)
    out_path = output_parquet_path.with_suffix(".parquet")
    df.write_parquet(out_path)
    log.info(f"Saved predictions to {out_path}")

import datetime as dt
from pathlib import Path
from typing import Any

from pydantic import BaseModel, FilePath
from torch.utils.data import DataLoader

from psycop.common.feature_generation.loaders.raw.load_ids import SplitName
from psycop.common.feature_generation.sequences.cohort_definer_to_prediction_times import (
    CohortToPredictionTimes,
)
from psycop.common.feature_generation.sequences.patient_loaders import (
    DiagnosisLoader,
    PatientLoader,
)
from psycop.common.sequence_models.aggregators import AggregationModule, AveragePooler
from psycop.common.sequence_models.dataset import PatientSlicesWithLabels
from psycop.common.sequence_models.tasks import (
    BEHRTForMaskedLM,
    EncoderForClassification,
)
from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    T2DCohortDefiner,
)


class DataCfg(BaseModel):
    lookbehind: dt.timedelta
    lookahead: dt.timedelta
    num_classes: int

    batch_size: int


class EncoderCfg(BaseModel):
    aggregation_module: AggregationModule
    optimizer_kwargs: dict[str, Any]
    lr_scheduler_kwargs: dict[str, Any]


class FinetuningConfig(BaseModel):
    checkpoint_path: FilePath  # This type ensures that the file exists

    encoder: EncoderCfg
    data: DataCfg


if __name__ == "__main__":
    encoder_cfg = EncoderCfg(
        aggregation_module=AveragePooler(),
        optimizer_kwargs={"lr": 1e-3},
        lr_scheduler_kwargs={"num_warmup_steps": 2, "num_training_steps": 10},
    )

    data_cfg = DataCfg(
        lookbehind=dt.timedelta(days=365),
        lookahead=dt.timedelta(days=365),
        num_classes=2,
        batch_size=32,
    )

    cfg = FinetuningConfig(
        checkpoint_path=Path("checkpoint.ckpt"),
        data=data_cfg,
        encoder=encoder_cfg,
    )

    patients = PatientLoader.get_split(
        event_loaders=[DiagnosisLoader(min_n_visits=5)],
        split=SplitName.TRAIN,
    )

    prediction_times = CohortToPredictionTimes(
        cohort_definer=T2DCohortDefiner(),
        patient_objects=patients,
    ).create_prediction_times(
        lookbehind=cfg.data.lookbehind,
        lookahead=cfg.data.lookahead,
    )

    patient_dataset_with_labels = PatientSlicesWithLabels(
        prediction_times=prediction_times,
    )

    loaded_model = BEHRTForMaskedLM.load_from_checkpoint(cfg.checkpoint_path)

    clf = EncoderForClassification(
        embedding_module=loaded_model.embedding_module,
        encoder_module=loaded_model.encoder_module,
        aggregation_module=cfg.encoder.aggregation_module,
        num_classes=cfg.data.num_classes,
        optimizer_kwargs=cfg.encoder.optimizer_kwargs,
        lr_scheduler_kwargs=cfg.encoder.lr_scheduler_kwargs,
    )

    dataloader = DataLoader(
        patient_dataset_with_labels,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        collate_fn=clf.collate_fn,
    )

    for input_ids, masked_labels in dataloader:
        output = clf(input_ids, masked_labels)
        loss = output["loss"]
        loss.backward()  # ensure that the backward pass works

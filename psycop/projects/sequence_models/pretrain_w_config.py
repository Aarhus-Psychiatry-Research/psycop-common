import logging
import sys
from pathlib import Path

project_path = Path(__file__).parents[3]
print(project_path)
sys.path.append(str(project_path))


from psycop.common.feature_generation.loaders.raw.load_ids import SplitName
from psycop.common.feature_generation.sequences.patient_loaders import (
    DiagnosisLoader, PatientLoader)
from psycop.common.sequence_models.dataset import PatientSliceDataset
from psycop.common.sequence_models.registry import Registry
from psycop.common.sequence_models.train import train


@Registry.datasets.register("diagnosis_only_patient_slice_dataset")
def create_patient_slice_dataset(
    min_n_visits: int,
    split_name: str,
) -> PatientSliceDataset:
    train_patients = PatientLoader.get_split(
        event_loaders=[
            DiagnosisLoader(min_n_visits=min_n_visits),
        ],
        split=SplitName(split_name),
    )
    return PatientSliceDataset([p.as_slice() for p in train_patients])


if __name__ == "__main__":
    config_path = Path(__file__).parent / "pretrain.cfg"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    )

    train(config_path)

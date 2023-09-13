from pathlib import Path

import pytest
from torch.utils.data import DataLoader, Dataset

from psycop.common.sequence_models.checkpoint_savers.base import (
    Checkpoint,
    TrainingState,
)
from psycop.common.sequence_models.checkpoint_savers.save_to_disk import (
    CheckpointToDisk,
)


class TestDataset(Dataset):
    def __init__(self):
        self.data = [(1, 2), (3, 4), (5, 6)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return self.data[idx]


@pytest.fixture()
def dataloader_for_tests() -> DataLoader:
    return DataLoader(TestDataset(), batch_size=1, shuffle=True)


def filename_is_numbered_with_steps(path: Path) -> bool:
    stem = path.stem

    if stem.find("_step_") == -1:
        return False
    return True


@pytest.mark.parametrize("override_on_save", [True, False])
def test_save_to_disk(
    tmp_path: Path,
    dataloader_for_tests: DataLoader,
    override_on_save: bool,
):
    saver = CheckpointToDisk(
        checkpoint_path=tmp_path,
        override_on_save=override_on_save,
    )

    checkpoint = Checkpoint(
        run_name="test",
        train_step=0,
        training_state=TrainingState(model_state_dict={}, optimizer_state_dict={}),
        loss=0.0,
        train_dataloader=dataloader_for_tests,
        val_dataloader=dataloader_for_tests,
    )

    saver.save(checkpoint)
    loaded_checkpoint = saver.load_latest()
    assert loaded_checkpoint is not None  # Check that the checkpoint was loaded

    for key, value in checkpoint.__dict__.items():
        if not isinstance(value, DataLoader):
            assert getattr(loaded_checkpoint, key) == value
        else:
            continue
            # TODO: Uncertain about how to test that the dataloaders' samplers
            # have the same state. By default, it uses a random sampler which appears
            # to have no real state

    # Test that the file is numbered with its step if override_on_save is False
    if not override_on_save:
        files = list(tmp_path.glob(r"*.pt"))
        unnumbered_matches = [file for file in files if file.stem == "test.pt"]
        assert len(unnumbered_matches) == 0

        numbered_matches = list(filter(filename_is_numbered_with_steps, files))
        assert len(numbered_matches) == 1
        assert numbered_matches[0].stem == "test_step_0"

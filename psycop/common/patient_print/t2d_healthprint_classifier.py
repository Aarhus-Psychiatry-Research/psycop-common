from pathlib import Path

import fastai.vision.all as vision
from fastai.data.transforms import get_image_files
from fastai.vision.augment import Resize
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner

from psycop.common.patient_print.healthprints_config import (
    HEALTHPRINT_PLOTS_DIR,
)


def example_labeller(filename: str) -> bool:
    return "positive" in filename


def test_train_patientprinter_classifier(plot_path: Path):
    files = get_image_files(plot_path)

    dls = ImageDataLoaders.from_name_func(
        plot_path,
        files,
        example_labeller,
    )

    learn = vision_learner(dls, vision.xresnet34, metrics=vision.error_rate)

    learn.fine_tune(100)


if __name__ == "__main__":
    test_train_patientprinter_classifier(plot_path=HEALTHPRINT_PLOTS_DIR)

from pathlib import Path

import fastai.vision.all as vision
from fastai.data.transforms import get_image_files
from fastai.vision.augment import Resize
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner


def diabetes_labeller(filename: str) -> bool:
    return "diabetes" in filename


def test_train_patientprinter_classifier():
    plots_base_path = Path("plots")
    files = get_image_files(plots_base_path)

    assert diabetes_labeller("diabetes_test.jpg") is True
    assert diabetes_labeller("control_test.jpg") is False

    dls = ImageDataLoaders.from_name_func(
        plots_base_path,
        files,
        diabetes_labeller,
        item_tfms=Resize(224, method="squish"),
    )

    learn = vision_learner(dls, vision.xresnet34, metrics=vision.error_rate)

    learn.fine_tune(1)

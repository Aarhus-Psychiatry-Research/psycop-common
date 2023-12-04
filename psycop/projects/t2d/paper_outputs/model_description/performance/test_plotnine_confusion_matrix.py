from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    ConfusionMatrix,
)
from psycop.common.model_evaluation.utils import TEST_PLOT_PATH
from psycop.projects.t2d.paper_outputs.model_description.performance.plotnine_confusion_matrix import (
    plotnine_confusion_matrix,
)


def test_plotnine_confusion_matrix():
    cm = ConfusionMatrix(
        true_positives=100_000,
        false_negatives=5000,
        false_positives=2000,
        true_negatives=2000,
    )

    plotnine_confusion_matrix(cm, outcome_text="T2D within 5 years").save(
        TEST_PLOT_PATH / "test_plotnine_confusion_matrix.png",
        width=5,
        height=5,
        dpi=600,
    )

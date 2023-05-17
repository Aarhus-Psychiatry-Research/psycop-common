from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    ConfusionMatrix,
)
from psycop.common.model_evaluation.utils import TEST_PLOT_PATH
from psycop.common.test_utils.str_to_df import str_to_df
from psycop.projects.t2d.paper_outputs.model_description.performance.plotnine_confusion_matrix import (
    plotnine_confusion_matrix,
)


def test_plotnine_confusion_matrix():
    cm = ConfusionMatrix(
        true_positives=10,
        false_negatives=5,
        false_positives=2,
        true_negatives=20,
    )

    plotnine_confusion_matrix(cm, x_title="Diabetes within 5 years").save(
        TEST_PLOT_PATH / "test_plotnine_confusion_matrix.png"
    )

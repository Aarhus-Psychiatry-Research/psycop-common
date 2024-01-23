from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    get_confusion_matrix_cells_from_df,
)
from psycop.common.test_utils.str_to_df import str_to_df


def test_get_confusion_matrix_cells_from_df():
    long_df = str_to_df(
        """true,pred,
        0,0, # 1 true negative
        1,0, # 2 false negative
        1,0,
        0,1, # 3 false positive
        0,1,
        0,1,
        1,1, # 4 true positive
        1,1,
        1,1,
        1,1
        """
    )

    cells = get_confusion_matrix_cells_from_df(long_df)

    assert cells.true_negatives == 1
    assert cells.false_negatives == 2
    assert cells.false_positives == 3
    assert cells.true_positives == 4

    assert cells.ppv == cells.true_positives / (cells.true_positives + cells.false_positives)
    assert cells.npv == cells.true_negatives / (cells.true_negatives + cells.false_negatives)
    assert cells.specificity == cells.true_negatives / (
        cells.true_negatives + cells.false_positives
    )
    assert cells.sensitivity == cells.true_positives / (
        cells.true_positives + cells.false_negatives
    )

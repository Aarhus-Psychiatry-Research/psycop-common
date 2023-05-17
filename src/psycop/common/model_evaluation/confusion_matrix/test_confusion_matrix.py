import pandas as pd
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    get_confusion_matrix_cells_from_long_df,
)
from psycop.common.test_utils.str_to_df import str_to_df


def create_long_confusion_matrix_df(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> pd.DataFrame:
    """Create a long confusion matrix dataframe."""

    df = pd.DataFrame(
        {
            "true": y_true,
            "pred": y_pred,
        },
    )

    cells = get_confusion_matrix_cells_from_long_df(df)

    ppv = f"PPV:\n{round(cells.ppv, 2)}"
    npv = f"NPV:\n{round(cells.npv, 2)}"
    sens = f"Sens:\n{round(cells.sensitivity, 2)}"
    spec = f"Spec:\n{round(cells.specificity, 2)}"

    rows = [
        {"true": "+", "pred": "+", "estimate": cells.true_positives},
        {"true": "+", "pred": "-", "estimate": cells.false_negatives},
        {"true": "+", "pred": "Agg", "estimate": sens},
        {"true": "-", "pred": "+", "estimate": cells.false_positives},
        {"true": "-", "pred": "-", "estimate": cells.true_negatives},
        {"true": "-", "pred": "Agg", "estimate": spec},
        {"true": "Agg", "pred": "1", "estimate": ppv},
        {"true": "Agg", "pred": "0", "estimate": npv},
    ]

    return pd.DataFrame(rows)


def test_create_long_confusion_matrix_df():
    input_df = str_to_df(
        """true,pred
1,1,
1,0,
1,0,
0,1,
0,1,
0,1,
0,0,
0,0,
0,0,
0,0,
""",
    )

    compute_df = pd.concat([input_df for _ in range(10)]).reset_index(drop=True)

    df = create_long_confusion_matrix_df(
        y_true=compute_df["true"],
        y_pred=compute_df["pred"],
    )

    assert df.shape == (8, 3)
    assert df.columns.tolist() == ["true", "pred", "estimate"]


def test_get_confusion_matrix_cells_from_long_df():
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

    cells = get_confusion_matrix_cells_from_long_df(long_df)

    assert cells.true_negatives == 1
    assert cells.false_negatives == 2
    assert cells.false_positives == 3
    assert cells.true_positives == 4

    assert cells.ppv == cells.true_positives / (
        cells.true_positives + cells.false_positives
    )
    assert cells.npv == cells.true_negatives / (
        cells.true_negatives + cells.false_negatives
    )
    assert cells.specificity == cells.true_negatives / (
        cells.true_negatives + cells.false_positives
    )
    assert cells.sensitivity == cells.true_positives / (
        cells.true_positives + cells.false_negatives
    )

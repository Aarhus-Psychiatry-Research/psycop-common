from dataclasses import dataclass

import pandas as pd


@dataclass
class ConfusionMatrix:
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    @property
    def ppv(self) -> float:
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def npv(self) -> float:
        return self.true_negatives / (self.true_negatives + self.false_negatives)

    @property
    def sensitivity(self) -> float:
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def specificity(self) -> float:
        return self.true_negatives / (self.true_negatives + self.false_positives)


def get_confusion_matrix_cells_from_df(df: pd.DataFrame) -> ConfusionMatrix:
    """Get confusion matrix cells from a long dataframe.

    Args:
        df (pd.DataFrame): Rows are observations, columns are "true" and "pred".
    """
    confusion_matrix_df = (
        df.groupby(["true", "pred"]).size().reset_index().rename(columns={0: "estimate"})
    )

    true_positives = confusion_matrix_df[
        (confusion_matrix_df["true"] == 1) & (confusion_matrix_df["pred"] == 1)
    ]["estimate"].values[0]
    true_negatives = confusion_matrix_df[
        (confusion_matrix_df["true"] == 0) & (confusion_matrix_df["pred"] == 0)
    ]["estimate"].values[0]

    false_positives = confusion_matrix_df[
        (confusion_matrix_df["true"] == 0) & (confusion_matrix_df["pred"] == 1)
    ]["estimate"].values[0]
    false_negatives = confusion_matrix_df[
        (confusion_matrix_df["true"] == 1) & (confusion_matrix_df["pred"] == 0)
    ]["estimate"].values[0]

    return ConfusionMatrix(
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )

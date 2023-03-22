import pandas as pd
from psycop_model_training.config_schemas.data import ColumnNamesSchema
from psycop_model_training.training.utils import create_eval_dataset


def test_create_eval_dataset():
    """Test that the eval dataset is created correctly."""
    col_names = ColumnNamesSchema(
        pred_timestamp="pred_timestamp",
        outcome_timestamp="outcome_timestamp",
        id="id",
        age="age",
        is_female="is_female",
        exclusion_timestamp="exclusion_timestamp",
        custom_columns=["custom1", "custom2"],
    )

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "y": [1, 0, 1, 0, 1],
            "y_hat_prob": [0.1, 0.2, 0.3, 0.4, 0.5],
            "y_hat_int": [0, 0, 0, 0, 1],
            "pred_timestamp": [1, 2, 3, 4, 5],
            "outcome_timestamp": [1, 2, 3, 4, 5],
            "age": [1, 2, 3, 4, 5],
            "is_female": [1, 2, 3, 4, 5],
            "exclusion_timestamp": [1, 2, 3, 4, 5],
            "custom1": [1, 2, 3, 4, 5],
            "custom2": [1, 2, 3, 4, 5],
        },
    )

    eval_dataset = create_eval_dataset(col_names=col_names, outcome_col_name="y", df=df)

    assert eval_dataset.custom_columns["custom1"].equals(df["custom1"])


if __name__ == "__main__":
    test_create_eval_dataset()

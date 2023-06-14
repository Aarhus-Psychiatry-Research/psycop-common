from care_ml.model_evaluation.config import TABLES_PATH
from care_ml.model_evaluation.dataset_description.utils import (
    load_feature_set,
    table_one_coercion,
    table_one_demographics,
)


def main():
    # load train and test splits
    df = load_feature_set()

    # create table one - demographics
    table_one_d = table_one_demographics(df)
    table_one_d.to_csv(
        TABLES_PATH.parent.parent.parent / "table_one_demographics.csv",
        index=False,
    )

    # create table one - coericon
    table_one_c = table_one_coercion(df)
    table_one_c.to_csv(
        TABLES_PATH.parent.parent.parent / "table_one_coercion.csv",
        index=False,
    )


if __name__ == "__main__":
    main()

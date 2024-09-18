import pandas as pd

from psycop.projects.t2d_bigdata.feature_generation.cohort_definition.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)
from psycop.projects.t2d_bigdata.feature_generation.cohort_definition.outcome_specification.medications import (
    get_first_antidiabetic_medication,
)
from psycop.projects.t2d_bigdata.feature_generation.cohort_definition.outcome_specification.t1d_diagnoses import (
    get_first_type_1_diabetes_diagnosis,
)
from psycop.projects.t2d.feature_generation.cohort_definition.outcome_specification.t2d_diagnoses import (
    get_first_type_2_diabetes_diagnosis,
)


def get_first_diabetes_indicator() -> pd.DataFrame:
    dfs = {
        "t1d_diagnoses": get_first_type_1_diabetes_diagnosis(),
        "t2d_diagnoses": get_first_type_2_diabetes_diagnosis(),
        "medications": get_first_antidiabetic_medication(),
        "lab_results": get_first_diabetes_lab_result_above_threshold(),
    }

    # Add a source column to each of the dfs
    for df_type in dfs:
        dfs[df_type]["source"] = df_type

    combined = pd.concat([dfs[k] for k in dfs], axis=0)

    first_diabetes_indicator = (
        combined.sort_values("timestamp").groupby("dw_ek_borger").first().reset_index(drop=False)
    )

    first_diabetes_indicator["value"] = 1
    return first_diabetes_indicator[["dw_ek_borger", "timestamp", "value", "source"]]


if __name__ == "__main__":
    df = get_first_diabetes_indicator()

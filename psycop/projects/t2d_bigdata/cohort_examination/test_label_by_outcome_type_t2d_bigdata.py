from dataclasses import dataclass

import polars as pl
import pytest

from psycop.projects.cvd.feature_generation.cohort_definition.outcome_specification.procedure_codes import (
    CVD_PROCEDURE_CODES,
)
from psycop.projects.t2d_bigdata.cohort_examination.label_by_outcome_type import (
    label_by_outcome_type,
)


# TODO: make general test
@dataclass(frozen=True)
class Ex:
    given: str
    then: str
    intention: str = ""


@pytest.mark.parametrize(
    ("example"),
    [
        Ex(given="KFNG05A", then="PCI"),
        Ex(given="A:DI691", then="Stroke"),
        Ex(given="A:DI509#+:AZAC2#+:ZDW10#B:DE109#B:DI214", then="MI"),
        Ex(given="A:DI639#+:AZAC2#B:DE780#B:DI109#B:DI489", then="Stroke"),
        Ex(given="A:DI21#+:KPEG", then="MI", intention="If overlapping, take the most severe."),
    ],
    ids=lambda ex: ex.given,
)
def test_grouping_by_outcome_type(example: Ex) -> None:
    df = pl.DataFrame({"diagnosis_code": [example.given]})

    result = label_by_outcome_type(
        df,
        procedure_col="diagnosis_code",
        type2procedures=CVD_PROCEDURE_CODES,  # TODO fh: make more relevant test
    )

    assert result.get_column("outcome_type").unique().to_list() == [example.then]

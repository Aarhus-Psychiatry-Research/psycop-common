from dataclasses import dataclass

import polars as pl
import pytest


def label_by_outcome_type(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    outcome2substrings = {
        "AMI": ["DI21", "DI22", "DI23"],
        "Stroke": ["DI6"],
        "PCI": ["KFNG"],
        "CABG": [
            "KFNA",
            "KFNB",
            "KFNC",
            "KFND",
            "KFNE",
            "KFNF",
            "KFNH",
            "KFNI",
            "KFNJ",
            "KFNK",
            "KFNW",
        ],
        "Coronary angiography": ["UXAC85"],
        "Intracranial endovascular thrombolysis": ["KAAL10", "KAAL11"],
        "Other intracranial endovascular surgery": ["KAAL99"],
        "A. iliaca": ["KPDE", "KPDF", "KPDH", "KPDM", "KPDP", "KPDQ"],
        "A. femoralis": ["KPEE", "KPEF", "KPEH", "KPEN", "KPEP", "KPEQ"],
        "A. poplitea and distal": [
            "KPFE",
            "KPFG",
            "KPFH",
            "KPFN",
            "KPFP",
            "KPFQ",
            "KPFQ",
            "KPDU74",
            "KPDU84",
            "KNFQ",
            "KNGQ",
            "KNHQ",
        ],
    }

    # Reverse to get the most severe outcome first
    # Initialise an empty column
    df = df.with_columns(pl.lit(None).alias("outcome_type"))
    for outcome, substrings in reversed(outcome2substrings.items()):
        for substring in substrings:
            df = df.with_columns(
                pl.when(pl.col(group_col).str.contains(substring))
                .then(pl.lit(outcome))
                .otherwise("outcome_type")
                .alias("outcome_type")
            )

    return df


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
        Ex(given="A:DA2+:KFNW", then="CABG"),
        Ex(given="A:DI509#+:AZAC2#+:ZDW10#B:DE109#B:DI214", then="AMI"),
        Ex(given="A:DI639#+:AZAC2#B:DE780#B:DI109#B:DI489", then="Stroke"),
        Ex(given="A:DI21#+:KPEG", then="AMI", intention="If overlapping, take the most severe."),
    ],
    ids=lambda ex: ex.given,
)
def test_grouping_by_outcome_type(example: Ex) -> None:
    df = pl.DataFrame({"diagnosis_code": [example.given]})

    result = label_by_outcome_type(df)

    assert result.get_column("outcome_type").unique().to_list() == [example.then]

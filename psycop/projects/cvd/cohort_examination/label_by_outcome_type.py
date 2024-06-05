import polars as pl


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
        "Iliac artery": ["KPDE", "KPDF", "KPDH", "KPDM", "KPDP", "KPDQ"],
        "Femoral artery": ["KPEE", "KPEF", "KPEH", "KPEN", "KPEP", "KPEQ"],
        "Popliteal artery and distal ": [
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

"""Quick and dirty script to quality check the SFI's and assess their usefullness
for the project"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from psycop.common.feature_generation.loaders.raw.load_text import load_all_notes

LOGFILE = Path("log.csv")
OPTIONS = "[G]ood sample, [B]ad sample, [Q]uit, [I]nput text"


@dataclass
class DfValues:
    sfi: str
    text: str
    index: int


def sample_sfi(df: pd.DataFrame) -> DfValues:
    """Sample a random sfi name and text from the data"""
    sample = df.sample(1)
    return DfValues(
        sfi=sample["overskrift"].item(),
        text=sample["value"].item(),
        index=sample.index.item(),
    )


def make_text_output(cur_sfi: str, cur_text: str) -> str:
    return f"""SFI: {cur_sfi}

{cur_text}

"""


def write_to_file(sfi: str, index: str, useful: bool, notes: str):
    if not LOGFILE.exists():
        LOGFILE.write_text("overskrift,index,useful,notes\n")
    with LOGFILE.open("a") as f:
        f.write(f"{sfi},{index}{useful},{notes}\n")


def main(df: pd.DataFrame):
    for index, row in df.iterrows():
        print(make_text_output(cur_sfi=row["overskrift"], cur_text=row["value"]))  # type: ignore
        print(OPTIONS)
        user_input = input("Input: ")
        if user_input == "g":
            write_to_file(
                sfi=row["overskrift"],  # type: ignore
                index=str(index),
                useful=True,
                notes="",
            )
        elif user_input == "b":
            write_to_file(
                sfi=row["overskrift"],  # type: ignore
                index=str(index),
                useful=False,
                notes="",
            )
        elif user_input == "q":
            break
        elif user_input == "i":
            notes = input("Input text: ")
            quality = input("Is the text [g]ood or [b]ad?")
            if quality == "g":
                write_to_file(
                    sfi=row["overskrift"],  # type: ignore
                    index=str(index),
                    useful=True,
                    notes=notes,
                )
            elif quality == "b":
                write_to_file(
                    sfi=row["overskrift"],  # type: ignore
                    index=str(index),
                    useful=False,
                    notes=notes,
                )
            else:
                print("Invalid input")
        else:
            print("Invalid input")


if __name__ == "__main__":
    SAMPLE_SIZE = 15

    DEV = False

    if DEV:
        DF = pd.DataFrame(
            {
                "text": ["test1", "test2", "test3"],
                "overskrift": ["test1", "test2", "test3"],
            },
        )
    else:
        DF = load_all_notes(include_sfi_name=True)  # type: ignore

    # sample an even number of texts from each overskrift
    samples = DF.groupby("overskrift").apply(lambda x: x.sample(SAMPLE_SIZE))
    # return to df form
    samples = samples.drop(columns=["overskrift"]).reset_index().set_index("level_1")
    main(df=samples)

"""Quick and dirty script to quality check the SFI's and assess their usefullness
for the project"""

from pathlib import Path
import polars as pl
from psycop.common.feature_generation.loaders.raw.load_text import load_all_notes

DEV = True

if DEV:
    DF = pl.DataFrame(
            {
                "text": ["test1", "test2", "test3"],
                "overskrift": ["test1", "test2", "test3"],
            }
        )
else:
    DF = pl.from_dataframe(load_all_notes(include_sfi_name=True)) # type: ignore

LOGFILE = Path("log.csv")
OPTIONS = "[G]ood sample, [B]ad sample, [Q]uit, [I]nput text"


def sample_sfi() -> tuple[str, str]:
    """Sample a random sfi name and text from the data"""
    sample = DF.sample(1)
    return sample["text"].item(), sample["overskrift"].item()


def make_text_output(cur_sfi: str, cur_text: str) -> str:
    return f"""SFI: {cur_sfi}

{cur_text}

"""


def write_to_file(cur_text: str, cur_sfi: str, useful: bool, notes: str):
    if not LOGFILE.exists():
        LOGFILE.write_text("text,overskrift,useful,notes\n")
    with open(LOGFILE, "a") as f:
        f.write(f"{cur_text},{cur_sfi},{useful},{notes}\n")

def main():
    while True:
        cur_text, cur_sfi = sample_sfi()
        print(make_text_output(cur_sfi=cur_sfi, cur_text=cur_text))
        print(OPTIONS)
        user_input = input("Input: ")
        if user_input == "g":
            write_to_file(cur_text=cur_text, cur_sfi=cur_sfi, useful=True, notes="")
        elif user_input == "b":
            write_to_file(cur_text=cur_text, cur_sfi=cur_sfi, useful=False, notes="")
        elif user_input == "q":
            break
        elif user_input == "i":
            notes = input("Input text: ")
            quality = input("Is the text [g]ood or [b]ad?")
            if quality == "g":
                write_to_file(cur_text=cur_text, cur_sfi=cur_sfi, useful=True, notes=notes)
            elif quality == "b":
                write_to_file(cur_text=cur_text, cur_sfi=cur_sfi, useful=False, notes=notes)
            else:
                print("Invalid input")
        else:
            print("Invalid input")

if __name__ == "__main__":
    main()



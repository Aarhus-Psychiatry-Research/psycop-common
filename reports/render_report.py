"""Tool to render a markdown document with figure and table numbering and added
captions.

Uses the keywords @title and @authors to add title and authors (might not really
be necessary but can be used for pretty styling). Add figures and tables using {filepath : caption}. Tables should
be in a dir called 'tables' and figures in a dir called 'figs'
"""


import pandas as pd

from psycopt2d.utils import PROJECT_ROOT

# import pandoc
# See comment in pyproject.toml on Pandoc, not currently in use. Should work now, see: https://github.com/boisgera/pandoc/pull/49#issuecomment-1265983279

AUTHORS = ["Lasse Hansen", "MB", "KCE"]
TITLE = "Paradigm Shattering Paper 1"


TABLE_DIR = PROJECT_ROOT / "tables"
PLOT_DIR = PROJECT_ROOT / "figs"
REPORT_PATH = PROJECT_ROOT / "reports"


def insert_figure(line: str, fig_number: int):
    """Add a figure to the report as md."""
    path, caption = line.split(":")
    path = path.strip().replace("{", "")
    caption = caption.strip().replace("}", "")

    figure = f"![Figure {fig_number}: {caption}]({path})\n"
    return figure


def insert_table(line: str, table_number: int):
    """Add a table to the report as md."""
    path, caption = line.split(":")
    path = path.strip().replace("{", "")
    caption = caption.strip().replace("}", "")
    md_table = csv_to_md(path)

    table = f"_Table {table_number}: {caption}_\n\n{md_table}\n"
    return table


def csv_to_md(filepath: str):
    """Load a table from a csv and return as markdown."""
    df = pd.read_csv(filepath)
    return df.to_markdown()


def center_text(text: list[str]):
    """Add html center align tags."""
    return "\n" + f"""<div align="center">{_join_by_newline(text)}</div>"""


def _join_by_newline(text: list[str]):
    """Join a list of strings by newline."""
    return "\n\n".join(text)


if __name__ == "__main__":
    with open(REPORT_PATH / "report.md") as f:  # pylint: disable=unspecified-encoding
        report = f.readlines()

    mdown_in_process = []
    FIG_COUNT = 1
    TABLE_COUNT = 1

    for line in report:
        if "@title" in line:
            line = line.replace("@title", TITLE)
        elif "@authors" in line:
            line = line.replace("@authors", center_text(AUTHORS))
        elif line.startswith("{"):
            if "figs" in line:
                line = insert_figure(line, fig_number=FIG_COUNT)
                FIG_COUNT += 1
            if "tables" in line:
                line = insert_table(line, table_number=TABLE_COUNT)
                TABLE_COUNT += 1
        mdown_in_process.append(line)

    MDOWN_OUT = "".join(mdown_in_process)
    # pandoc_format = pandoc.read(md)
    # pandoc.write(pandoc_format, file=str(REPORT_PATH / "auto_report.docx"))

    with open(  # pylint: disable=unspecified-encoding
        REPORT_PATH / "auto_report.md",
        "w",
    ) as f:
        f.write(MDOWN_OUT)

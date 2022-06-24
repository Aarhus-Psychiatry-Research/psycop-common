"""Small tool for automatic report generation.

Insert filenames of tables (.csv files), names of figures, and headers +
text
"""
from pathlib import Path

import pandas as pd
import pandoc
from wasabi import MarkdownRenderer

md = MarkdownRenderer()

TABLE_DIR = Path() / "tables"
PLOT_DIR = Path() / "figs"
REPORT_PATH = Path() / "reports"


def load_table(filepath: str):
    df = pd.read_csv(TABLE_DIR / filepath)
    return df.to_markdown()


def load_figure(filepath: str, caption: str):
    return f"![{caption}]({str(PLOT_DIR / filepath)})"


## TODO
# Set defaults for tables and figures - which ones we normally include
# Handlers for numbers in captions or paragraphs
toc = {
    "table1.csv": "Caption for table 1",
    "table2.csv": "Caption for table 2",
    "## Some section": "Text to put under the heading",
    "fig1.png": "Fig1 caption",
    "fig2.png": "Fig2 caption",
}

for key, text in toc.items():
    if key.startswith("table"):
        md.add(load_table(key))
    elif key.startswith("fig"):
        md.add(load_figure(key, text))
    elif key.startswith("#"):
        level = key.count("#")
        title = key.strip("# ")
        md.add(md.title(level, title))

    md.add(text)


pandoc_format = pandoc.read(md.text)
pandoc.write(pandoc_format, file=REPORT_PATH / "auto_report.docx")

with open(REPORT_PATH / "auto_report.md", "w") as f:
    f.write(md.text)

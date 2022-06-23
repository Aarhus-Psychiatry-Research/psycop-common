from pathlib import Path

import pandas as pd
from wasabi import MarkdownRenderer

md = MarkdownRenderer()

TABLE_DIR = Path() / "tables"
PLOT_DIR = Path() / "figs"


def load_table(filepath: str):
    df = pd.read_csv(TABLE_DIR / filepath)
    return df.to_markdown()


def load_figure_and_caption(filepath: str, caption: str):
    return f"![{caption}]({str(PLOT_DIR / filepath)})"


# Set default tables (auc by group, sensitivity thresholds...)
tables = {"table1.csv": "Caption for table 1", "table2.csv": "Caption for table 2"}

# Set default figures
figures = {
    "performance_by_blabla.png": "Fig1 caption",
    "performance2.png": "Fig2 caption",
}

for fig, caption in tables.items():
    md.add(load_table(fig))
    md.add(caption)

for fig, caption in figures.items():
    md.add(load_figure_and_caption(fig, caption))
    md.add(caption)

md.text()

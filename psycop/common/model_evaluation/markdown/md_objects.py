import abc
import copy
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import plotnine as pn
import polars as pl


class MarkdownArtifact(ABC):
    title: str

    @abc.abstractmethod
    def get_markdown(self) -> str:
        raise NotImplementedError


@dataclass
class MarkdownFigure(MarkdownArtifact):
    title: str
    description: str
    file_path: Path
    title_prefix: str = "Figure"
    relative_to_path: Optional[Path] = None

    def __post_init__(self):
        if not self.file_path.exists():
            raise FileNotFoundError

        if self.relative_to_path is not None:
            self.file_path = self.file_path.relative_to(self.relative_to_path)

    def get_markdown(self) -> str:
        return f"""{self.title_prefix} {self.title}

![]({self.file_path.as_posix()})

{self.description}
"""


@dataclass
class MarkdownTable(MarkdownArtifact):
    title: str
    description: str
    table: pd.DataFrame
    title_prefix: str = "Table"

    @classmethod
    def from_filepath(
        cls: type["MarkdownTable"],
        table_path: Path,
        title: str,
        description: str,
        title_prefix: str = "Table",
    ) -> "MarkdownTable":
        if table_path.suffix == ".csv":
            table = pd.read_csv(table_path)
        else:
            table = pd.read_excel(table_path)

        return cls(title=title, description=description, title_prefix=title_prefix, table=table)

    def get_markdown(self) -> str:
        return f"""{self.title_prefix} {self.title}

{self.table.to_markdown(index=False)}

{self.description}
"""


def create_supplementary_from_markdown_artifacts(
    artifacts: Sequence[MarkdownArtifact],
    first_table_index: int = 1,
    table_title_prefix: str = "eTable",
    first_figure_index: int = 1,
    figure_title_prefix: str = "eFigure",
) -> str:
    markdown = """# Supplementary material

"""

    figure_index = first_figure_index
    table_index = first_table_index

    for artifact in artifacts:
        # Create a copy to avoid modifying the pointed-to object, e.g.
        # to avoid adding two titles if the function is called twice on
        # the same object
        artifact_copy = copy.copy(artifact)
        artifact_type = type(artifact_copy)

        if artifact_type == MarkdownTable:
            artifact_copy.title = (
                f"## **{table_title_prefix} {table_index}**: {artifact_copy.title}"
            )
            table_index += 1
        elif artifact_type == MarkdownFigure:
            artifact_copy.title = (
                f"## **{figure_title_prefix} {figure_index}**: {artifact_copy.title}"
            )
            figure_index += 1
        else:
            raise ValueError(
                f"Artifact type {artifact_type} not supported. "
                f"Only MarkdownTable and MarkdownFigure are supported."
            )

        markdown += artifact_copy.get_markdown()

    return markdown

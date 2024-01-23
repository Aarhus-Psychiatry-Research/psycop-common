import abc
import copy
from abc import ABC
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import pandas as pd


class MarkdownArtifact(ABC):
    def __init__(
        self, title: str, file_path: Path, description: str, check_filepath_exists: bool = True
    ):
        self.title = title
        self.file_path = file_path
        self.description = description

        if check_filepath_exists and not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} does not exist")

    @abc.abstractmethod
    def get_markdown(self) -> str:
        raise NotImplementedError


class MarkdownFigure(MarkdownArtifact):
    def __init__(
        self,
        file_path: Path,
        description: str,
        title: str,
        title_prefix: str = "Figure",
        check_filepath_exists: bool = True,
        relative_to_path: Optional[Path] = None,
    ):
        super().__init__(
            title=title,
            file_path=file_path,
            description=description,
            check_filepath_exists=check_filepath_exists,
        )

        self.title_prefix = title_prefix

        if relative_to_path is not None:
            self.file_path = self.file_path.relative_to(relative_to_path)

    def get_markdown(self) -> str:
        return f"""{self.title_prefix} {self.title}

![]({self.file_path.as_posix()})

{self.description}
"""


class MarkdownTable(MarkdownArtifact):
    def __init__(
        self,
        title: str,
        file_path: Path,
        description: str,
        title_prefix: str = "Table",
        check_filepath_exists: bool = True,
    ):
        super().__init__(
            title=title,
            file_path=file_path,
            description=description,
            check_filepath_exists=check_filepath_exists,
        )
        self.title_prefix = title_prefix

    def _get_table_as_pd(self) -> pd.DataFrame:
        if self.file_path.suffix == ".csv":
            return pd.read_csv(self.file_path)

        if self.file_path.suffix == ".xlsx":
            return pd.read_excel(self.file_path)

        raise ValueError(
            f"File extension {self.file_path.suffix} not supported. "
            f"Only .csv and .xlsx are supported."
        )

    def get_markdown_table(self) -> str:
        df = self._get_table_as_pd()
        return df.to_markdown(index=False)  # type: ignore

    def get_markdown(self) -> str:
        return f"""{self.title_prefix} {self.title}

{self.get_markdown_table()}

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

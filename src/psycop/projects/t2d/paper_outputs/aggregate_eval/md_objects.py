from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import pandas as pd


class MarkdownArtifact:
    def __init__(
        self,
        title: str,
        file_path: Path,
        description: str,
        check_filepath_exists: bool = True,
    ):
        self.title = title
        self.file_path = file_path
        self.description = description

        if check_filepath_exists and not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} does not exist")


class MarkdownFigure(MarkdownArtifact):
    def __init__(
        self,
        file_path: Path,
        description: str,
        title: str,
        title_prefix: str = "Figure",
        check_filepath_exists: bool = True,
        relative_to: Optional[Path] = None,
    ):
        super().__init__(
            title=title,
            file_path=file_path,
            description=description,
            check_filepath_exists=check_filepath_exists,
        )
        self.title_prefix = title_prefix

        if relative_to is not None:
            self.file_path = self.file_path.relative_to(relative_to)

    def get_markdown(self) -> str:
        return f"""{self.title}

![{self.title}]({self.file_path.as_posix()})

{self.description}
"""


class MarkdownTable(MarkdownArtifact):
    def __init__(
        self,
        file_path: Path,
        description: str,
        title: str,
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
            return pd.read_excel(
                self.file_path,
            )

        raise ValueError(
            f"File extension {self.file_path.suffix} not supported. "
            f"Only .csv and .xlsx are supported.",
        )

    def get_markdown_table(self) -> str:
        df = self._get_table_as_pd()
        return df.to_markdown(index=False)

    def get_markdown(self) -> str:
        return f"""{self.title}

{self.get_markdown_table()}

{self.description}
"""


def create_supplementary_from_markdown_artifacts(
    artifacts: Sequence[MarkdownArtifact],
    first_table_index: int = 1,
    first_figure_index: int = 1,
) -> str:
    tables = tuple(a for a in artifacts if isinstance(a, MarkdownTable))
    figures = tuple(a for a in artifacts if isinstance(a, MarkdownFigure))

    markdown = """# Supplementary material

"""

    for artifacts in figures, tables:
        if isinstance(artifacts[0], MarkdownTable):
            title_prefix = "eTable"
            index = first_table_index
        elif isinstance(artifacts[0], MarkdownFigure):
            title_prefix = "eFigure"
            index = first_figure_index
        else:
            raise ValueError(f"Unknown artifact type {type(artifacts[0])}")

        for artifact in artifacts:
            artifact.title = f"## **{title_prefix} {index}**: {artifact.title}"
            markdown += f"""{artifact.get_markdown()}


"""
            index += 1

    return markdown

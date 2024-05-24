from pathlib import Path

import pytest

from psycop.common.model_evaluation.markdown.md_objects import (
    MarkdownFigure,
    MarkdownTable,
    create_supplementary_from_markdown_artifacts,
)
from psycop.common.test_utils.str_to_df import str_to_df


class TestMarkdownFigure:
    def test_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            MarkdownFigure(title="Title", file_path=Path("path/to/file"), description="Description")

    def test_markdown_figure_output(self):
        output = MarkdownFigure(
            title="Figure_title", file_path=Path("path/to/file"), description="Description"
        ).get_markdown()

        assert isinstance(output, str)


class TestMarkdownTable:
    table_csv = str_to_df(
        """col1,col2,col3
1,2,3,
4,5,6,
"""
    )

    def test_creating_markdown_table(self, tmp_path: Path):
        self.table_csv.to_csv(tmp_path / "table.csv", index=False)

        md_table = MarkdownTable.from_filepath(
            title="Table_title", table_path=tmp_path / "table.csv", description="Description"
        )

        md = md_table.table.to_markdown(index=False)

        assert isinstance(md, str)
        assert "|---" in md
        assert "--:|" in md
        assert md[-1] == "|"
        assert md[0] == "|"


class TestCreateSupplementaryFromMarkdownArtifacts:
    table_csv = str_to_df(
        """col1,col2,col3
1,2,3,
4,5,6,
"""
    )

    figure_string = "Figure here"

    def test_can_output_markdown(self, tmp_path: Path):
        self.table_csv.to_csv(tmp_path / "table.csv", index=False)
        filepath = tmp_path / "test.md"
        with filepath.open("w") as f:
            f.write("Testing 123")

        artifacts = [
            MarkdownTable.from_filepath(
                title="Table_title", table_path=tmp_path / "table.csv", description="Description"
            ),
            MarkdownFigure(
                title="Figure_title",
                file_path=filepath,
                description="Figure description",
                relative_to_path=tmp_path,
            ),
            MarkdownTable.from_filepath(
                title="Table_title", table_path=tmp_path / "table.csv", description="Description"
            ),
            MarkdownFigure(
                title="Figure_title",
                file_path=Path("path/to/file"),
                description="Figure description",
            ),
        ]

        md = create_supplementary_from_markdown_artifacts(
            artifacts=artifacts,
            first_table_index=3,
            table_title_prefix="eTable",
            first_figure_index=1,
            figure_title_prefix="eFigure",
        )

        assert "eTable 3" in md
        assert "eFigure 1" in md

        with (tmp_path / "test.md").open("w") as f:
            f.write(md)

        print(f"Test file exists at {tmp_path / 'test.md'}")

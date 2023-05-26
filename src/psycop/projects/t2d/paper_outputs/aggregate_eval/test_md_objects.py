from pathlib import Path

import pandas as pd
import pytest
from psycop.common.test_utils.str_to_df import str_to_df
from psycop.projects.t2d.paper_outputs.aggregate_eval.md_objects import (
    MarkdownFigure,
    MarkdownTable,
    create_supplementary_from_markdown_artifacts,
)


class TestMarkdownFigure:
    def test_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            MarkdownFigure(
                title="Title",
                file_path=Path("path/to/file"),
                description="Description",
            )

    def test_markdown_figure_output(self):
        output = MarkdownFigure(
            title="Figure_title",
            file_path=Path("path/to/file"),
            description="Description",
            check_filepath_exists=False,
        ).get_markdown()

        assert isinstance(output, str)


class TestMarkdownTable:
    table_csv = str_to_df(
        """col1,col2,col3
1,2,3,
4,5,6,
""",
    )

    def test_loading_of_table(self, tmp_path: Path):
        self.table_csv.to_csv(tmp_path / "table.csv", index=False)

        md_table = MarkdownTable(
            title="Table_title",
            file_path=tmp_path / "table.csv",
            description="Description",
            check_filepath_exists=True,
        )

        dataframe = md_table._get_table_as_pd()

        assert isinstance(dataframe, pd.DataFrame)

    def test_creating_markdown_table(self, tmp_path: Path):
        self.table_csv.to_csv(tmp_path / "table.csv", index=False)

        md_table = MarkdownTable(
            title="Table_title",
            file_path=tmp_path / "table.csv",
            description="Description",
            check_filepath_exists=True,
        )

        md = md_table.get_markdown_table()

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
""",
    )

    figure_string = "Figure here"

    def test_can_output_markdown(self, tmp_path: Path):
        self.table_csv.to_csv(tmp_path / "table.csv", index=False)

        artifacts = [
            MarkdownTable(
                title="Table_title",
                file_path=tmp_path / "table.csv",
                description="Description",
                check_filepath_exists=False,
            ),
            MarkdownFigure(
                title="Figure_title",
                file_path=Path("path/to/file"),
                description="Figure description",
                check_filepath_exists=False,
            ),
            MarkdownTable(
                title="Table_title",
                file_path=tmp_path / "table.csv",
                description="Description",
                check_filepath_exists=False,
            ),
            MarkdownFigure(
                title="Figure_title",
                file_path=Path("path/to/file"),
                description="Figure description",
                check_filepath_exists=False,
            ),
        ]

        md = create_supplementary_from_markdown_artifacts(
            artifacts=artifacts,
            first_table_index=3,
            first_figure_index=1,
        )

        with (tmp_path / "test.md").open("w") as f:
            f.write(md)

        print(f"Test file exists at {tmp_path / 'test.md'}")

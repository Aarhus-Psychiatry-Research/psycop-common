from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class MarkdownArtifact:
    title: str
    file_path: Path
    description: str


@dataclass
class MarkdownFigure(MarkdownArtifact):
    def get_markdown(self) -> str:
        return f"""### {self.title}
![{self.title}]({self.file_path})
{self.description}
"""


@dataclass
class MarkdownTable(MarkdownArtifact):
    def _get_table_as_pd(self) -> pd.DataFrame:
    
    def create_markdown_table(self) -> str:
        
    
    def get_markdown(self) -> str
        return f"""### {self.title}
{self.table}
{self.description}
"""

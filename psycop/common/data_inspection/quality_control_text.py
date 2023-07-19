"""CLI application for quality control of text data."""

from typing import Type

import polars as pl
from textual._path import CSSPathType
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer
from textual.driver import Driver
from textual.keys import Keys
from textual.reactive import Reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    Placeholder,
    Static,
    Welcome,
)

from psycop.common.feature_generation.loaders.raw.load_text import load_all_notes

DF = pl.DataFrame(
        {
            "text": ["test1", "test2", "test3"],
            "overskrift": ["test1", "test2", "test3"],
        }
    )


def sample_sfi() -> tuple[str, str]:
    """Sample a random sfi name and text from the data"""
    sample = DF.sample(1)
    return sample["text"].item(), sample["overskrift"].item()


def make_text_output() -> str:
    cur_text, cur_sfi = sample_sfi()
    return f"""## SFI: {cur_sfi}

{cur_text}"""

TEST_MD = TEXT = """
[b]Set your background[/b]
[@click=set_background('red')]Red[/]
[@click=set_background('green')]Green[/]
[@click=set_background('blue')]Blue[/]
"""


class TextQualityControl(App):
    """A Textual app to quality control text"""
    cur_md: str =  "test"
    #make_text_output()
    
    BINDINGS = [("d", "toggle_dark_mode", "Toggle dark mode"), ("n", "new_sample", "New sample")]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Container(Static(TEST_MD))
        yield Footer()
        #yield ScrollableContainer(TextQuality())

    def action_toggle_dark_mode(self) -> None:
        print("toggle dark mode")
        self.dark = not self.dark
    def action_new_sample(self) -> None:
        print("get new sample")
        self.cur_md = make_text_output()

    def action_set_background(self, color: str) -> None:
        self.screen.styles.background = color

if __name__ == "__main__":


    TextQualityControl().run()


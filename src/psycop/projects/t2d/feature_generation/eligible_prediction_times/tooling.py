from dataclasses import dataclass

import polars as pl


@dataclass
class StepDelta:
    step_name: str
    n_before: int
    n_after: int

    @property
    def n_dropped(self) -> int:
        return self.n_before - self.n_after


stepdeltas = []


def add_stepdelta_manual(step_name: str, n_before: int, n_after: int) -> None:
    stepdeltas.append(
        StepDelta(
            step_name=step_name,
            n_before=n_before,
            n_after=n_after,
        ),
    )


def add_stepdelta_from_df(
    step_name: str,
    before_df: pl.DataFrame,
    after_df: pl.DataFrame,
) -> None:
    stepdeltas.append(
        StepDelta(
            step_name=step_name,
            n_before=before_df.shape[0],
            n_after=after_df.shape[0],
        ),
    )

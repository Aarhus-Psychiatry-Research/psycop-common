import polars as pl

from psycop.common.cohort_definition import StepDelta

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

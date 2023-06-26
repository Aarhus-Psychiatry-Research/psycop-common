


from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl
from pydantic import BaseModel



@dataclass
class StepDelta:
    step_name: str
    n_before: int
    n_after: int

    @property
    def n_dropped(self) -> int:
        return self.n_before - self.n_after


class FilteredCohort(BaseModel):
    """A filtered cohort is a cohort that has been filtered by a set of criteria."""
    cohort: pl.DataFrame
    filter_steps: list[StepDelta]



class Cohort(ABC):
    """Interface for a cohort definition."""
    @abstractmethod
    def get_eligible_prediction_times(self) -> FilteredCohort:
        ...

    @abstractmethod
    def get_outcome_timestamps(self) -> pl.DataFrame:
        ...

    




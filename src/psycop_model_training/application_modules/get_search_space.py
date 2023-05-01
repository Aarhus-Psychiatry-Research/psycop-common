import random

import pandas as pd
from psycop_model_training.config_schemas.basemodel import BaseModel
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.utils.col_name_inference import (
    infer_look_distance,
    infer_outcome_col_name,
)
from wasabi import Printer


class TrainerSpec(BaseModel):
    """Specification for starting a trainer.

    Provides overrides for the config file.
    """

    lookahead_days: int
    model_name: str


class SearchSpaceInferrer:
    """Infer the search space for the model training pipeline."""

    def __init__(
        self,
        cfg: FullConfigSchema,
        train_df: pd.DataFrame,
        model_names: list[str],
    ):
        self.cfg = cfg
        self.train_df = train_df
        self.model_names = model_names

    def _get_impossible_lookaheads(
        self,
        potential_lookaheads: list[int],
    ) -> list[int]:
        """Some look_ahead and look_behind distances will result in 0 valid
        prediction times.

        E.g. if we only have 4 years of data:
        - min_lookahead = 2 years
        - min_lookbehind = 3 years

        Will mean that no rows satisfy the criteria.
        """
        max_interval_days = (
            max(self.train_df[self.cfg.data.col_name.pred_timestamp])
            - min(
                self.train_df[self.cfg.data.col_name.pred_timestamp],
            )
        ).days

        msg = Printer(timestamp=True)
        lookaheads_without_rows: list[int] = [
            dist for dist in potential_lookaheads if dist > max_interval_days
        ]

        if lookaheads_without_rows:
            msg.info(
                f"Not fitting model to {lookaheads_without_rows}, since no rows satisfy the criteria.",
            )

        return lookaheads_without_rows

    def _get_possible_lookaheads(self) -> list[int]:
        """Some look_ahead and look_behind distances will result in 0 valid
        prediction times. Only return combinations which will allow some
        prediction times.

        E.g. if we only have 4 years of data:
        - min_lookahead = 2 years
        - min_lookbehind = 3 years

        Will mean that no rows satisfy the criteria.
        """
        outcome_col_names = infer_outcome_col_name(
            df=self.train_df,
            allow_multiple=True,
            prefix=self.cfg.data.outc_prefix,
        )

        potential_lookaheads: list[int] = [
            int(dist) for dist in infer_look_distance(col_name=outcome_col_names)
        ]

        impossible_lookaheads = self._get_impossible_lookaheads(
            potential_lookaheads=potential_lookaheads,
        )

        return list(set(potential_lookaheads) - set(impossible_lookaheads))

    def _combine_lookaheads_and_model_names_to_trainer_specs(
        self,
        possible_lookahead_days: list[int],
    ) -> list[TrainerSpec]:
        """Generate trainer specs for all combinations of lookaheads and model
        names."""
        msg = Printer(timestamp=True)

        random.shuffle(possible_lookahead_days)

        if self.model_names:
            msg.warn(
                "model_names was specified in train_models_for_each_cell_in_grid, overriding self.cfg.model.name",
            )

        model_name_queue = self.model_names if self.model_names else self.cfg.model.name

        # Create all combinations of lookahead_days and models
        trainer_combinations_queue = [
            TrainerSpec(lookahead_days=lookahead_days, model_name=model_name)
            for lookahead_days in possible_lookahead_days.copy()
            for model_name in model_name_queue
        ]

        return trainer_combinations_queue

    def get_trainer_specs(self) -> list[TrainerSpec]:
        """Get all possible combinations of lookaheads and models."""
        return self._combine_lookaheads_and_model_names_to_trainer_specs(
            possible_lookahead_days=self._get_possible_lookaheads(),
        )

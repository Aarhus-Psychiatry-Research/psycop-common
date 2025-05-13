import logging
from dataclasses import dataclass

import plotnine as pn

from psycop.projects.ect.model_evaluation.auroc_by.auroc_by_view import auroc_by_view
from psycop.projects.ect.model_evaluation.auroc_by.time_from_first_visit_model import (
    TimeFromFirstVisitDF,
    auroc_by_time_from_first_visit_model,
)
from psycop.projects.ect.model_evaluation.single_run_artifact import SingleRunPlot


@dataclass
class AUROCByTimeFromFirstVisitPlot(SingleRunPlot):
    data: TimeFromFirstVisitDF

    def __call__(self) -> pn.ggplot:
        logging.info(f"Starting {self.__class__.__name__}")

        return auroc_by_view(
            df=self.data.to_pandas(),
            x_column="unit_from_event_binned",
            line_y_col_name="auroc",
            xlab="Months from first visit",
        )


if __name__ == "__main__":

    import polars as pl

    from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits_to_psychiatry
    from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
        EvalFrame,
    )
    from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk

    experiment = f"ECT-hparam-structured_only-xgboost-no-lookbehind-filter"
    experiment_path = f"E:/shared_resources/ect/eval_runs/{experiment}_best_run_evaluated_on_test"
    experiment_df = read_eval_df_from_disk(experiment_path)
    eval_frame = EvalFrame(frame=experiment_df, allow_extra_columns=True)
    all_visits_df=pl.from_pandas(physical_visits_to_psychiatry())

    AUROCByTimeFromFirstVisitPlot(
        auroc_by_time_from_first_visit_model(eval_frame=eval_frame, all_visits_df=all_visits_df)
        ).__call__

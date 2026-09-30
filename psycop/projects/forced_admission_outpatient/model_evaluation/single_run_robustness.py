import logging
from pathlib import Path
from typing import TYPE_CHECKING

import patchworklib as pw
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits_to_psychiatry
from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame
from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.forced_admission_outpatient.model_evaluation.auroc_by.age_model import (
    auroc_by_age_model,
)
from psycop.projects.forced_admission_outpatient.model_evaluation.auroc_by.age_view import (
    AUROCByAge,
)
from psycop.projects.forced_admission_outpatient.model_evaluation.auroc_by.quarter_model import (
    auroc_by_quarter_model,
)
from psycop.projects.forced_admission_outpatient.model_evaluation.auroc_by.quarter_view import (
    AUROCByQuarterPlot,
)
from psycop.projects.forced_admission_outpatient.model_evaluation.auroc_by.sex_model import (
    auroc_by_sex_model,
)
from psycop.projects.forced_admission_outpatient.model_evaluation.auroc_by.sex_view import (
    AUROCBySex,
)
from psycop.projects.forced_admission_outpatient.model_evaluation.auroc_by.time_from_first_visit_model import (
    auroc_by_time_from_first_visit_model,
)
from psycop.projects.forced_admission_outpatient.model_evaluation.auroc_by.time_from_first_visit_view import (
    AUROCByTimeFromFirstVisitPlot,
)
from psycop.projects.forced_admission_outpatient.model_evaluation.single_run_artifact import (
    SingleRunPlot,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import plotnine as pn

log = logging.getLogger(__name__)


def single_run_robustness(
    eval_frame: EvalFrame,
    sex_df: pl.DataFrame,
    birthdays: pl.DataFrame,
    all_visits_df: pl.DataFrame,
    save_dir: Path,
) -> pw.Bricks:
    eval_df = eval_frame.frame

    plots: Sequence[SingleRunPlot] = [
        AUROCBySex(auroc_by_sex_model(eval_df=eval_df, sex_df=sex_df)),
        AUROCByAge(
            auroc_by_age_model(eval_df=eval_df, birthdays=birthdays, bins=[18, *range(20, 80, 10)])
        ),
        AUROCByTimeFromFirstVisitPlot(
            auroc_by_time_from_first_visit_model(eval_frame=eval_frame, all_visits_df=all_visits_df)  # type: ignore
        ),
        AUROCByQuarterPlot(auroc_by_quarter_model(eval_frame=eval_frame)),  # type: ignore
    ]

    ggplots: list[pn.ggplot] = []
    for plot in plots:
        log.info(f"Starting processing of {plot.__class__.__name__}")
        ggplots.append(plot())

    figure = create_patchwork_grid(plots=ggplots, single_plot_dimensions=(5, 4.5), n_in_row=2)

    output_path = save_dir / "auroc_robustness_plot.png"
    figure.savefig(output_path)

    return figure


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    import polars as pl

    from psycop.common.feature_generation.loaders.raw.load_visits import (
        physical_visits_to_psychiatry,
    )
    from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame
    from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk

    experiment = "ia_outpatient_all_features_training"
    experiment_path = f"E:/shared_resources/forced_admissions_outpatient/eval_runs/{experiment}_best_run_evaluated_on_test"
    eval_df = read_eval_df_from_disk(experiment_path)

    eval_df = eval_df.with_columns(
        [pl.col("y").cast(pl.Int64), pl.col("y_hat_prob").cast(pl.Float64)]
    )

    eval_frame = EvalFrame(frame=eval_df, allow_extra_columns=True)

    all_visits_df = pl.from_pandas(physical_visits_to_psychiatry())

    sex_df = pl.from_pandas(sex_female())

    birthday = pl.from_pandas(birthdays())

    single_run_robustness(
        eval_frame=eval_frame,
        birthdays=birthday,
        sex_df=sex_df,
        all_visits_df=all_visits_df,
        save_dir=Path(f"{experiment_path}/figures"),
    )

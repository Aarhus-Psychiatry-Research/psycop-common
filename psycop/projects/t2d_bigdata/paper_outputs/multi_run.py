from collections.abc import Sequence
from pathlib import Path

import coloredlogs

import psycop.projects.t2d_bigdata.model_evaluation.multi_run.auroc_by_run_data as abrd
from psycop.projects.t2d_bigdata.model_evaluation.multi_run import auroc_by_run_presentation


def multi_run_facade(output_path: Path, runs: Sequence[abrd.RunSelector]) -> None:
    model = abrd.model(runs=runs)

    table = auroc_by_run_presentation.table(model)
    table.write_csv(output_path / "Model comparisons.csv")

    plot = auroc_by_run_presentation.plot(model)
    plot.save(output_path / "Model comparisons.png", limitsize=False, dpi=300, width=10, height=5)


if __name__ == "__main__":
    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    runs = [
        abrd.RunSelector(experiment_name="T2D-bigdata", run_name="T2D-bigdata 1, base, XGB"),
        abrd.RunSelector(
            experiment_name="T2D-bigdata", run_name="T2D-bigdata 1, (mean, min, max), XGB"
        ),
        abrd.RunSelector(
            experiment_name="T2D-bigdata", run_name="T2D-bigdata 1, lookbehind: 90,365,730, XGB"
        ),
        abrd.RunSelector(experiment_name="T2D-bigdata", run_name="T2D-bigdata, SCORE2"),
        abrd.RunSelector(experiment_name="T2D-bigdata, h, l-1, XGB", run_name="Best"),
        abrd.RunSelector(experiment_name="T2D-bigdata, h, l-2, XGB", run_name="Best"),
        abrd.RunSelector(experiment_name="T2D-bigdata, h, l-3, XGB", run_name="Best"),
        abrd.RunSelector(experiment_name="T2D-bigdata, h, l-4, XGB", run_name="Best"),
        abrd.RunSelector(experiment_name="T2D-bigdata, h, l-1, logR", run_name="Best"),
        abrd.RunSelector(experiment_name="T2D-bigdata, h, l-2, logR", run_name="Best"),
        abrd.RunSelector(experiment_name="T2D-bigdata, h, l-3, logR", run_name="Best"),
        abrd.RunSelector(experiment_name="T2D-bigdata, h, l-4, logR", run_name="Best"),
    ]
    multi_run_facade(output_path=Path(), runs=runs)

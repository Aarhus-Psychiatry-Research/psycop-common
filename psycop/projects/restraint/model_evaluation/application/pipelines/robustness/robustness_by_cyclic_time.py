from pathlib import Path

import plotnine as pn

from psycop.common.model_evaluation.binary.time.periodic_data import (
    roc_auc_by_periodic_time_df,
)
from psycop.projects.restraint.model_evaluation.config import (
    BEST_DEV_RUN,
    COLOURS,
    MODEL_NAME,
    PN_THEME,
    ROBUSTNESS_PATH,
    TEXT_EVAL_RUN,
    TEXT_ROBUSTNESS_PATH,
)
from psycop.projects.restraint.utils.best_runs import Run


def auroc_by_day_of_week(run: Run, path: Path):
    eval_ds = run.get_eval_dataset()

    binned_df = roc_auc_by_periodic_time_df(
        labels=eval_ds.y,  # type: ignore
        y_hat_probs=eval_ds.y_hat_probs,  # type: ignore
        timestamps=eval_ds.pred_timestamps,
        bin_period="D",
        n_bootstraps=1000,
    )

    binned_df["proportion_in_bin"] = (
        binned_df["n_in_bin"] / sum(binned_df.n_in_bin) * 5.65
    )
    binned_df["n_in_bin"] = binned_df["n_in_bin"].astype(int)

    (
        pn.ggplot(
            binned_df,
            pn.aes("time_bin", "auroc"),
        )
        + pn.geom_bar(
            pn.aes(x="time_bin", y="proportion_in_bin"),
            stat="identity",
            position="identity",
            fill=COLOURS["blue"],
        )
        + pn.geom_path(group=1, size=0.5)
        + pn.geom_text(
            pn.aes(y="proportion_in_bin", label="n_in_bin"),
            va="bottom",
            size=11,
        )
        + pn.geom_pointrange(
            pn.aes(ymin="ci_lower", ymax="ci_upper"),
            size=0.5,
        )
        + pn.labs(
            x="Day of the week",
            y="AUROC",
            title=f"{MODEL_NAME[run.name]} Performance Across Week",
        )
        + pn.coord_cartesian(ylim=(0.75, 0.97))
        + PN_THEME
        + pn.theme(axis_text_x=pn.element_text(rotation=90))
    ).save(
        path / "auc_by_day_of_the_week.png",
    )


def auroc_by_month_of_year(run: Run, path: Path):
    eval_ds = run.get_eval_dataset()

    binned_df = roc_auc_by_periodic_time_df(
        labels=eval_ds.y,  # type: ignore
        y_hat_probs=eval_ds.y_hat_probs,  # type: ignore
        timestamps=eval_ds.pred_timestamps,
        bin_period="M",
        n_bootstraps=1000,
    )

    binned_df["proportion_in_bin"] = (
        binned_df["n_in_bin"] / sum(binned_df.n_in_bin) * 8.2
    )
    binned_df["n_in_bin"] = binned_df["n_in_bin"].astype(int)

    (
        pn.ggplot(
            binned_df,
            pn.aes("time_bin", "auroc"),
        )
        + pn.geom_bar(
            pn.aes(x="time_bin", y="proportion_in_bin"),
            stat="identity",
            position="identity",
            fill=COLOURS["blue"],
        )
        + pn.geom_path(group=1, size=0.5)
        + pn.geom_text(
            pn.aes(y="proportion_in_bin", label="n_in_bin"),
            va="bottom",
            size=11,
        )
        + pn.geom_pointrange(
            pn.aes(ymin="ci_lower", ymax="ci_upper"),
            size=0.5,
        )
        + pn.labs(
            x="Month of the year",
            y="AUROC",
            title=f"{MODEL_NAME[run.name]} Performance Across Year",
        )
        + pn.coord_cartesian(ylim=(0.61, 0.97))
        + PN_THEME
        + pn.theme(axis_text_x=pn.element_text(rotation=90))
    ).save(
        path / "auc_by_month_of_the_year.png",
    )


if __name__ == "__main__":
    auroc_by_day_of_week(BEST_DEV_RUN, ROBUSTNESS_PATH)
    auroc_by_day_of_week(TEXT_EVAL_RUN, TEXT_ROBUSTNESS_PATH)
    auroc_by_month_of_year(BEST_DEV_RUN, ROBUSTNESS_PATH)
    auroc_by_month_of_year(TEXT_EVAL_RUN, TEXT_ROBUSTNESS_PATH)

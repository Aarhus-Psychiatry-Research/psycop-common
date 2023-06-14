from pathlib import Path

import plotnine as pn
from care_ml.model_evaluation.config import (
    COLOURS,
    EVAL_RUN,
    MODEL_NAME,
    PN_THEME,
    ROBUSTNESS_PATH,
    TEXT_EVAL_RUN,
    TEXT_ROBUSTNESS_PATH,
)
from care_ml.utils.best_runs import Run
from psycop.common.model_evaluation.binary.time.absolute_data import (
    create_roc_auc_by_absolute_time_df,
)


def roc_auc_by_calendar_time(run: Run, path: Path):
    eval_ds = run.get_eval_dataset()

    binned_df = create_roc_auc_by_absolute_time_df(
        labels=eval_ds.y,
        y_hat_probs=eval_ds.y_hat_probs,
        timestamps=eval_ds.pred_timestamps,
        bin_period="Y",
        confidence_interval=True,
        n_bootstraps=1000,
    )

    binned_df["proportion_in_bin"] = binned_df["n_in_bin"] / sum(binned_df.n_in_bin) * 5
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
            x="Year",
            y="AUROC",
            title=f"{MODEL_NAME[run.name]} Performance Over Time",
        )
        + pn.coord_cartesian(ylim=(0.4, 0.95))
        + PN_THEME
    ).save(
        path / "auc_by_calendar_time.png",
    )


if __name__ == "__main__":
    roc_auc_by_calendar_time(EVAL_RUN, ROBUSTNESS_PATH)
    roc_auc_by_calendar_time(TEXT_EVAL_RUN, TEXT_ROBUSTNESS_PATH)

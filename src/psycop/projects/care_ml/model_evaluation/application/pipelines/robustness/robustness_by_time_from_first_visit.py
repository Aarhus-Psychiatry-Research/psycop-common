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
from care_ml.model_evaluation.data.load_true_data import (
    load_eval_df,
)
from care_ml.utils.best_runs import Run
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_auroc_by_timedelta_df,
)


def roc_auc_by_time_from_first_prediction(run: Run, path: Path):
    df = load_eval_df(
        wandb_group=run.group.name,
        wandb_run=run.name,
    )

    df["first_prediction_day"] = df.groupby("adm_id")["pred_timestamps"].transform(
        "min",
    )

    binned_df = get_auroc_by_timedelta_df(
        y=df.y,
        y_hat_probs=df.y_hat_probs,
        time_one=df.first_prediction_day,
        time_two=df.pred_timestamps,
        direction="t2-t1",
        bins=[*range(0, 55, 5)],
        bin_unit="D",
        n_bootstraps=100000,
    )

    binned_df["proportion_in_bin"] = binned_df["n_in_bin"] / sum(binned_df.n_in_bin) * 3
    binned_df["n_in_bin"] = binned_df["n_in_bin"].astype(int)

    (
        pn.ggplot(
            binned_df,
            pn.aes("unit_from_event_binned", "auroc"),
        )
        + pn.geom_bar(
            pn.aes(x="unit_from_event_binned", y="proportion_in_bin"),
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
            x="Days in admission",
            y="AUROC",
            title=f"{MODEL_NAME[run.name]} Performance Over Admission",
        )
        + PN_THEME
    ).save(
        path / "auc_by_time_from_first_pred_day_in_adm.png",
    )


if __name__ == "__main__":
    roc_auc_by_time_from_first_prediction(EVAL_RUN, ROBUSTNESS_PATH)
    roc_auc_by_time_from_first_prediction(TEXT_EVAL_RUN, TEXT_ROBUSTNESS_PATH)

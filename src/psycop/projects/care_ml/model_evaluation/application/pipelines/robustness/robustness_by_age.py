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
from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df


def roc_auc_by_age(run: Run, path: Path):
    eval_ds = run.get_eval_dataset()

    binned_df = get_auroc_by_input_df(
        eval_dataset=eval_ds,  # type: ignore
        input_values=eval_ds.age,  # type: ignore
        input_name="age",
        bins=[18, *range(20, 80, 5)],
        n_bootstraps=1000,
    )

    binned_df["proportion_in_bin"] = (
        binned_df["n_in_bin"] / sum(binned_df.n_in_bin) * 7.5
    )
    binned_df["n_in_bin"] = binned_df["n_in_bin"].astype(int)

    (
        pn.ggplot(
            binned_df,
            pn.aes("age_binned", "auroc"),
        )
        + pn.geom_bar(
            pn.aes(x="age_binned", y="proportion_in_bin"),
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
            x="Age",
            y="AUROC",
            title=f"{MODEL_NAME[run.name]} Performance by Age",
        )
        + pn.coord_cartesian(ylim=(0.3, 1.05))
        + PN_THEME
        + pn.theme(axis_text_x=pn.element_text(rotation=90))
    ).save(
        path / "auc_by_age.png",
        dpi=300,
    )


if __name__ == "__main__":
    roc_auc_by_age(EVAL_RUN, ROBUSTNESS_PATH)
    roc_auc_by_age(TEXT_EVAL_RUN, TEXT_ROBUSTNESS_PATH)

from pathlib import Path

import pandas as pd
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


def roc_auc_by_sex(run: Run, path: Path):
    eval_ds = run.get_eval_dataset()

    binned_df = get_auroc_by_input_df(
        eval_dataset=eval_ds,  # type: ignore
        input_values=eval_ds.is_female,  # type: ignore
        input_name="is_female",
        bin_continuous_input=False,
        n_bootstraps=1000,
    )

    patients = (
        pd.DataFrame({"ids": eval_ds.ids, "sex": eval_ds.is_female})
        .drop_duplicates()
        .sex.value_counts()
    )

    binned_df["proportion_in_bin"] = (
        binned_df["n_in_bin"] / sum(binned_df.n_in_bin) * 1.7
    )

    binned_df = binned_df.merge(patients, left_on="is_female", right_on=patients.index)
    binned_df["is_female"] = binned_df["is_female"].replace({0: "Male", 1: "Female"})
    binned_df["n_in_bin"] = binned_df["n_in_bin"].astype(int)

    (
        pn.ggplot(
            binned_df,
            pn.aes("is_female", "auroc"),
        )
        + pn.geom_bar(
            pn.aes(x="is_female", y="proportion_in_bin"),
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
        )
        + pn.labs(
            x="Sex",
            y="AUROC",
            title=f"{MODEL_NAME[run.name]} Performance by Sex",
        )
        + pn.coord_cartesian(ylim=(0.75, 0.95))
        + PN_THEME
    ).save(
        path / "auc_by_sex.png",
    )


if __name__ == "__main__":
    roc_auc_by_sex(EVAL_RUN, ROBUSTNESS_PATH)
    roc_auc_by_sex(TEXT_EVAL_RUN, TEXT_ROBUSTNESS_PATH)

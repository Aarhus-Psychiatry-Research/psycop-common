import pandas as pd
import plotnine as pn
from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.t2d.paper_outputs.config import COLORS, PN_THEME
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_age import (
    plot_robustness,
)


def test_auroc_by_age(synth_eval_dataset: EvalDataset):
    df = get_auroc_by_input_df(
        eval_dataset=synth_eval_dataset,
        input_values=synth_eval_dataset.age,  # type: ignore
        input_name="age",
        bins=[18, *range(20, 80, 50)],
        confidence_interval=True,
    )

    df["proportion_of_n"] = df["n_in_bin"] / df["n_in_bin"].sum()

    plot_robustness(df)


def new_func(df: pd.DataFrame) -> pn.ggplot:
    p = (
        pn.ggplot(df, pn.aes(x="age_binned", y="auroc"))
        + pn.geom_bar(
            pn.aes(x="age_binned", y="proportion_of_n"),
            stat="identity",
            fill=COLORS.background,
        )
        + pn.geom_pointrange(
            pn.aes(ymin="ci_lower", ymax="ci_upper"),
            color=COLORS.primary,
            size=0.5,
        )
        + pn.geom_path(group=1, color=COLORS.primary, size=1)
        + pn.xlab("Age")
        + pn.ylab("AUROC / Proportion of patients")
        + PN_THEME
    )

    return p

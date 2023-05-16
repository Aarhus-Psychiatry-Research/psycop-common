import pandas as pd
import plotnine as pn
from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.projects.t2d.paper_outputs.config import COLORS, EVAL_RUN, PN_THEME
from psycop.projects.t2d.utils.best_runs import Run


def plot_robustness(df: pd.DataFrame) -> pn.ggplot:
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


def auroc_by_age(run: Run) -> pn.ggplot:
    print("Plotting AUC by age")
    eval_ds = run.get_eval_dataset()

    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.age,  # type: ignore
        input_name="age",
        bins=[18, *range(20, 80, 10)],
        bin_continuous_input=True,
        confidence_interval=True,
    )

    return plot_robustness(df)


if __name__ == "__main__":
    auroc_by_age(run=EVAL_RUN)

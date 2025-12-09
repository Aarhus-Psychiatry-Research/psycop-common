import plotnine as pn

from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.clozapine.model_eval.performance.robustness.clozapine_robustness_plot import (
    clozapine_plot_robustness,
)


def clozapine_auroc_by_sex(eval_ds: EvalDataset) -> pn.ggplot:
    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.is_female,  # type: ignore
        input_name="is_female",
        confidence_interval=True,
        bin_continuous_input=False,
    )

    bool_to_sex = {False: "Male", True: "Female"}
    df["Sex"] = df["is_female"].map(bool_to_sex).astype("category")

    return clozapine_plot_robustness(df, x_column="Sex", line_y_col_name="auroc", xlab="Sex")

import plotnine as pn

from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_plot import (
    t2d_plot_robustness,
)


def scz_bp_auroc_by_age(eval_ds: EvalDataset) -> pn.ggplot:
    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.age,  # type: ignore
        input_name="age",
        bins=[18, *range(20, 80, 10)],
        bin_continuous_input=True,
        confidence_interval=True,
    )

    p = t2d_plot_robustness(
        df,
        x_column="age_binned",
        line_y_col_name="auroc",
        xlab="Age",
        rotate_x_axis_labels_degrees=45,
    )

    return p

from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.t2d_extended.model_evaluation.model_description.robustness.robustness_plot import (
    t2d_extended_plot_robustness,
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

    t2d_extended_plot_robustness(df, x_column="age_binned", line_y_col_name="auroc", xlab="Age")

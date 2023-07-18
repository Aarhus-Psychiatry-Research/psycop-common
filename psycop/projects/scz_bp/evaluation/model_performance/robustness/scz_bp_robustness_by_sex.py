import plotnine as pn

from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_plot import (
    t2d_plot_robustness,
)

from psycop.projects.scz_bp.evaluation.pipeline_objects import PipelineRun


def scz_bp_auroc_by_sex(run: PipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset(custom_columns=["is_female"])

    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.is_female,  # type: ignore
        input_name="is_female",
        confidence_interval=True,
        bin_continuous_input=False,
    )

    int_to_sex = {0: "Male", 1: "Female"}
    df["Sex"] = df["is_female"].map(int_to_sex).astype("category")

    return t2d_plot_robustness(
        df,
        x_column="Sex",
        line_y_col_name="auroc",
        xlab="Sex",
    )


if __name__ == "__main__":
    from psycop.projects.scz_bp.evaluation.model_selection.performance_by_group_lookahead_model_type import (
        DEVELOPMENT_PIPELINE_RUN,
    )

    scz_bp_auroc_by_sex(run=DEVELOPMENT_PIPELINE_RUN)

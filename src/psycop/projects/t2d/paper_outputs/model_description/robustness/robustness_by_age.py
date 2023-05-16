from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN
from psycop.projects.t2d.utils.best_runs import Run


def auroc_by_age(run: Run):
    print("Plotting AUC by age")
    eval_ds = run.get_eval_dataset()

    get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.age,  # type: ignore
        input_name="age",
        bins=[18, *range(20, 80, 10)],
        bin_continuous_input=True,
        confidence_interval=True,
    )

    # TODO: Create plot function


if __name__ == "__main__":
    auroc_by_age(run=EVAL_RUN)

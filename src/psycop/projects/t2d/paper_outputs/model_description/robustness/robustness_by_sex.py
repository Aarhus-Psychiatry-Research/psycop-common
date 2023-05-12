from psycop.model_evaluation.binary.subgroups.sex import plot_roc_auc_by_sex
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, ROBUSTNESS_PATH


def roc_auc_by_sex():
    print("Plotting AUC by sex")
    eval_ds = EVAL_RUN.get_eval_dataset(custom_columns=["is_female"])

    plot_roc_auc_by_sex(
        eval_dataset=eval_ds,
        save_path=ROBUSTNESS_PATH / "auc_by_sex.png",
    )


# %%
if __name__ == "__main__":
    roc_auc_by_sex()

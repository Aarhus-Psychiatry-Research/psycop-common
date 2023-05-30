from psycop.projects.t2d.paper_outputs.config import EVAL_RUN
from psycop.projects.t2d.utils.best_runs import ModelRun


def roc_auc_by_sex(run: ModelRun):
    print("Plotting AUC by sex")
    run.get_eval_dataset(custom_columns=["is_female"])

    # TODO: Create plot


# %%
if __name__ == "__main__":
    roc_auc_by_sex(run=EVAL_RUN)

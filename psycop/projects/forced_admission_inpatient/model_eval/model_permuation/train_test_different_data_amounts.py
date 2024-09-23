"""Experiments with models trained and evaluated using only part of the data"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from psycop.common.model_training.preprocessing.post_split.pipeline import (
    create_post_split_pipeline,
)
from psycop.common.model_training.utils.col_name_inference import get_col_names
from psycop.projects.forced_admission_inpatient.model_eval.model_description.performance.cross_validation_performance_table import (
    crossvalidate,
    load_data,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def train_model_on_different_training_data_amounts(
    run: ForcedAdmissionInpatientPipelineRun,
    save: bool = True,
    fractions: Sequence[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
) -> pd.DataFrame:
    """Train a single model and evaluate it."""
    cfg = run.inputs.cfg

    dataset = load_data(cfg)

    outcome_col_name_for_train, train_col_names = get_col_names(cfg, dataset)

    pipe = create_post_split_pipeline(cfg)

    roc_aucs = []
    cfs = []
    std_devs = []
    oof_intervals = []
    train_aurocs = []

    # sample 10, 25, 50, 75, 100 percent of the data and train the model on it
    for i in fractions:
        fraction_dataset = (
            dataset.copy().sample(frac=i, random_state=1).reset_index(drop=True)
            if i != 1.0
            else dataset.copy()
        )

        eval_dataset, oof_aucs, train_aucs = crossvalidate(
            cfg=cfg,
            train=fraction_dataset,
            pipe=pipe,
            outcome_col_name=outcome_col_name_for_train,  # type: ignore
            train_col_names=train_col_names,
        )
        roc_auc = roc_auc_score(  # type: ignore
            eval_dataset.y, eval_dataset.y_hat_probs
        )

        high_oof_auc = max(oof_aucs)
        low_oof_auc = min(oof_aucs)
        train_auroc = np.mean(train_aucs)  # type: ignore
        mean_auroc = np.mean(oof_aucs)
        std_dev_auroc = np.std(oof_aucs)  # type: ignore
        std_error_auroc = std_dev_auroc / np.sqrt(len(oof_aucs))
        dof = len(oof_aucs) - 1
        conf_interval = stats.t.interval(0.95, dof, loc=mean_auroc, scale=std_error_auroc)

        roc_aucs.append(roc_auc)
        cfs.append(f"[{round(conf_interval[0],3)}:{round(conf_interval[1],3)}]")
        std_devs.append(round(std_dev_auroc, 3))
        oof_intervals.append(f"{low_oof_auc}-{high_oof_auc}")
        train_aurocs.append(train_auroc)

    df_dict = {
        "Data fraction": fractions,
        "Apparent AUROC score": train_aurocs,
        "Internal AUROC score": roc_aucs,
        "95 percent confidence interval": cfs,
        "Standard deviation": std_devs,
        "5-fold out-of-fold AUROC interval": oof_intervals,
    }
    df = pd.DataFrame(df_dict)

    output_path = run.paper_outputs.paths.tables / "performance_by_training_fraction.xlsx"

    if save:
        df.to_excel(output_path, index=False)

    return df


def plot_performance_by_amount_of_training_data(run: ForcedAdmissionInpatientPipelineRun) -> None:
    df = train_model_on_different_training_data_amounts(run=run, save=False)

    # Plot a line graph of the performance by amount of training data
    fig, ax = plt.subplots()
    ax.plot(df["Data fraction"], df["Internal AUROC score"], label="Internal AUROC score")
    ax.plot(df["Data fraction"], df["Apparent AUROC score"], label="Apparent AUROC score")
    ax.fill_between(
        df["Data fraction"],
        df["Internal AUROC score"] - df["Standard deviation"],
        df["Internal AUROC score"] + df["Standard deviation"],
        alpha=0.2,
    )
    ax.fill_between(
        df["Data fraction"],
        df["Internal AUROC score"] - df["Standard deviation"],
        df["Internal AUROC score"] + df["Standard deviation"],
        alpha=0.2,
    )
    ax.set_xlabel("Data fraction")
    ax.set_ylabel("AUROC score")

    plt.legend()

    # Save the plot
    output_path = (
        run.paper_outputs.paths.figures / "fa_inpatient_performance_by_training_fraction.png"
    )
    plt.savefig(output_path)

    plt.show()


if __name__ == "__main__":
    from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    plot_performance_by_amount_of_training_data(run=get_best_eval_pipeline())

    train_model_on_different_training_data_amounts(
        run=get_best_eval_pipeline(), fractions=[0.1, 0.25, 0.5, 0.75, 1.0]
    )

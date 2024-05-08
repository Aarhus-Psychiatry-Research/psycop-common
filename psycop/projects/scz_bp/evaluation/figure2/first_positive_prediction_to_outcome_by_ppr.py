"""Table with models as rows and time from first positive to outcome at different 
positive predicted rates as columns, stratified by BP and SCZ"""


from pathlib import Path
from typing import Sequence

import pandas as pd

from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR
from psycop.projects.scz_bp.evaluation.figure2.first_positive_prediction_to_outcome import (
    scz_bp_first_pred_to_event_stratified,
)
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_best_run_in_experiment,
)


def scz_bp_time_from_first_positive_prediction_to_outcome_table(
    modality2experiment: dict[str, str], pprs: Sequence[float], group_by_outcome: bool
) -> pd.DataFrame:
    modality_dfs: list[pd.DataFrame] = []
    for modality, experiment in modality2experiment.items():
        eval_ds = scz_bp_get_eval_ds_from_best_run_in_experiment(experiment_name=experiment)
        ppr_dfs: list[pd.DataFrame] = []
        for ppr in pprs:
            ppr_df = scz_bp_first_pred_to_event_stratified(eval_ds=eval_ds, ppr=ppr).df
            ppr_df["ppr"] = ppr
            ppr_dfs.append(ppr_df)
        modality_dfs.append(pd.concat(ppr_dfs).assign(modality=modality))

    if group_by_outcome:
        df = (
            pd.concat(modality_dfs)
            .groupby(["modality", "ppr", "outcome"])
            .agg({"years_from_pred_to_event": "median"})
            .round(1)
        )
        df_wide = df.reset_index().pivot(
            index=["modality", "outcome"], columns="ppr", values="years_from_pred_to_event"
        )
    else:
        df = (
            pd.concat(modality_dfs)
            .groupby(["modality", "ppr"])
            .agg({"years_from_pred_to_event": "median"})
            .round(1)
        )
        df_wide = df.reset_index().pivot(
            index=["modality"], columns="ppr", values="years_from_pred_to_event"
        )

    df_wide = df_wide.reset_index()
    df_wide["modality"] = pd.Categorical(
        df_wide["modality"], categories=list(modality2experiment.keys())
    )

    return df_wide.sort_values("modality")


if __name__ == "__main__":
    # add PPV 

    modality2experiment_mapping = modality2experiment = {
        "Structured + text + synthetic": "sczbp/structured_text_xgboost_ddpm",
        "Structured + text": "sczbp/structured_text_xgboost",
        "Structured only ": "sczbp/structured_only-xgboost",
        "Text only": "sczbp/tfidf_1000-xgboost",
    }
    pprs = [0.01, 0.02, 0.04, 0.06, 0.08]

    table = scz_bp_time_from_first_positive_prediction_to_outcome_table(
        modality2experiment=modality2experiment_mapping, pprs=pprs, group_by_outcome=False
    )

    with open(SCZ_BP_EVAL_OUTPUT_DIR / "figure_2_a.html", "w") as f:
        f.write(table.to_html())
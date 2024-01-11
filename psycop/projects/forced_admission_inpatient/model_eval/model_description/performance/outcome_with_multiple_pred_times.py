import pandas as pd
import plotnine as pn
from wasabi import Printer

from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)

msg = Printer(timestamp=True)


def get_prediction_times_with_outcome_shared_by_n_other(
    eval_dataset: EvalDataset,
    n: int,
): 
    df = pd.DataFrame(
        {
            "id": eval_dataset.ids,
            "y": eval_dataset.y,
            "pred_timestamps": eval_dataset.pred_timestamps,
            "outcome_timestamps": eval_dataset.outcome_timestamps,
        },
    )

    df = df.dropna(subset=['outcome_timestamps'])

    df['outcome_uuid'] = df['id'].astype(str) + df['outcome_timestamps'].astype(str)

    # Count occurrences of each outcome_uuid
    outcome_uuid_counts = df['outcome_uuid'].value_counts()

    # Filter the DataFrame based on the count
    filtered_df = df[df['outcome_uuid'].isin(outcome_uuid_counts[outcome_uuid_counts == n].index)]

    print(filtered_df.head())


if __name__ == "__main__":
    from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    run = get_best_eval_pipeline()
    eval_dataset = run.pipeline_outputs.get_eval_dataset()
    get_prediction_times_with_outcome_shared_by_n_other(eval_dataset, n)
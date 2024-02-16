import pandas as pd
import plotnine as pn
from IPython.display import display
from wasabi import Printer

from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.forced_admission_inpatient.model_eval.config import FA_PN_THEME
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)

msg = Printer(timestamp=True)


def _get_prediction_times_with_outcome_shared_by_n_other(
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

    return filtered_df


def plot_distribution_of_n_pred_times_per_outcome(
        run: ForcedAdmissionInpatientPipelineRun,
        eval_dataset: EvalDataset,
        max_n: int,
    ) -> pn.ggplot:

    dist = [_get_prediction_times_with_outcome_shared_by_n_other(eval_dataset, n).outcome_uuid.nunique() for n in range(1, max_n)]
    
    df = pd.DataFrame({'n_pred_times': [category for category, count in zip(range(1, max_n), dist) for _ in range(count)]})

    n_pred_times_counts = df['n_pred_times'].value_counts()
    filtered_df = df[df['n_pred_times'].isin(n_pred_times_counts[n_pred_times_counts > 4].index)]
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df['n_pred_times'] = filtered_df['n_pred_times'].astype(str)

    plot = (
    pn.ggplot(filtered_df) 
    + FA_PN_THEME
    + pn.geom_bar(pn.aes(x='n_pred_times'), fill='#009E73')
    + pn.labs(x='No. prediction times per outcome', y='Count')
    + pn.annotate('text', x=3.5, y=70, label='Counts < 5 have been removed', size=10, color='red') 
    )

    auroc_path = run.paper_outputs.paths.figures / "fa_n_pred_times_per_outcome_distribution.png"
    plot.save(auroc_path)

    return plot



if __name__ == "__main__":
    from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    run = get_best_eval_pipeline()
    eval_dataset = run.pipeline_outputs.get_eval_dataset()
    max_n=20
    plot_distribution_of_n_pred_times_per_outcome(run, eval_dataset, max_n)
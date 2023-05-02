from psycop.model_evaluation.base_charts import plot_basic_chart
from psycop.model_evaluation.binary.time.timedelta_data import (
    create_sensitivity_by_time_to_outcome_df,
)
from psycop.model_evaluation.binary.time.timedelta_plots import (
    plot_sensitivity_by_time_until_diagnosis,
)
from psycop.model_evaluation.utils import TEST_PLOT_PATH
from psycop.model_training.training_output.dataclasses import EvalDataset


def test_overlay_barplot(subsampled_eval_dataset: EvalDataset):
    plot_sensitivity_by_time_until_diagnosis(
        eval_dataset=subsampled_eval_dataset,
        y_title="Sensitivity",
        confidence_interval=0.95,
        save_path=TEST_PLOT_PATH / "test_overlay_barplot.png",
    )


def test_plot_bar_chart(synth_eval_dataset: EvalDataset):
    plot_df = create_sensitivity_by_time_to_outcome_df(
        eval_dataset=synth_eval_dataset,
        outcome_timestamps=synth_eval_dataset.outcome_timestamps,
        prediction_timestamps=synth_eval_dataset.pred_timestamps,
        desired_positive_rate=0.5,
    )
    plot_basic_chart(
        x_values=plot_df["days_to_outcome_binned"],  # type: ignore
        y_values=plot_df["sens"],  # type: ignore
        x_title="Days to outcome",
        y_title="Sensitivity",
        plot_type="bar",
        save_path=TEST_PLOT_PATH / "test_plot_basic_chart.png",
    )

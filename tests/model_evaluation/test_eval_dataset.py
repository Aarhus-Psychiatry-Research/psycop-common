from psycop_model_training.model_eval.dataclasses import EvalDataset


def test_plot_time_from_first_positive_to_event(synth_eval_dataset: EvalDataset):
    synth_eval_dataset.to_df()

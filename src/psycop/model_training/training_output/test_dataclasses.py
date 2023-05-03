from psycop.model_training.training_output.dataclasses import EvalDataset


def test_eval_dataset_direction_of_positives_for_quantile(
    subsampled_eval_dataset: EvalDataset,
):
    (
        positives_twenty_percent,
        _,
    ) = subsampled_eval_dataset.get_predictions_for_positive_rate(0.8)
    (
        positives_zero_percent,
        _,
    ) = subsampled_eval_dataset.get_predictions_for_positive_rate(0.0)

    assert positives_twenty_percent.mean() > positives_zero_percent.mean()

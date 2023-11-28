from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    get_true_positives,
)
from psycop.common.model_evaluation.binary.performance_by_ppr.prop_of_all_events_hit_by_true_positive import (
    get_percentage_of_events_captured,
    get_prop_of_events_captured_from_eval_dataset,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.common.test_utils.str_to_df import str_to_df


def test_get_percentage_of_events_captured_from_eval_dataset(
    subsampled_eval_dataset: EvalDataset,
):
    get_prop_of_events_captured_from_eval_dataset(
        eval_dataset=subsampled_eval_dataset,
        positive_rate=0.02,
    )


def test_get_percentage_of_events_captured():
    input_df = str_to_df(
        """id,y,pred
        1,1,0, # Not captured
        2,1,0, # Not captured
        3,1,1, # Captured
        3,1,1, # Not relevant: ID is 3
        4,0,0, # Not relevant: y is 0
        """,
    )

    prop_of_events_captured = get_percentage_of_events_captured(df=input_df)

    assert prop_of_events_captured == (1 / 3)


def test_get_true_positives(synth_eval_dataset: EvalDataset):
    get_true_positives(eval_dataset=synth_eval_dataset, positive_rate=0.02)

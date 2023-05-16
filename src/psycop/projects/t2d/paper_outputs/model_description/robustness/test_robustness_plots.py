import pandas as pd
from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.common.model_training.training_output.dataclasses import EvalDataset


def test_auroc_by_age(synth_eval_dataset: EvalDataset):
    df = get_auroc_by_input_df(
        eval_dataset=synth_eval_dataset,
        input_values=synth_eval_dataset.age,  # type: ignore
        input_name="age",
        bins=[18, *range(20, 80, 50)],
        confidence_interval=True,
    )

    pass

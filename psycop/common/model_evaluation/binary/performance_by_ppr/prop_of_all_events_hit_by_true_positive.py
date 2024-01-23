import pandas as pd

from psycop.common.model_training.training_output.dataclasses import EvalDataset


def get_prop_of_events_captured_from_eval_dataset(
    eval_dataset: EvalDataset, positive_rate: float
) -> float:
    df = pd.DataFrame(
        {
            "y": eval_dataset.y,
            "pred": eval_dataset.get_predictions_for_positive_rate(positive_rate)[0],
            "id": eval_dataset.ids,
            "age": eval_dataset.age,
        }
    )

    return get_percentage_of_events_captured(df=df)


def get_percentage_of_events_captured(df: pd.DataFrame) -> float:
    # Get all patients with at least one event and at least one positive prediction

    # TODO: #44 How do we handle if a patient can have more than one event?
    # Then we kind of need something like "event-id" to see whether each event was captured or not.
    df_patients_with_events = (
        df.groupby("id").filter(lambda x: x["y"].sum() > 0).groupby("id").head(1)  # type: ignore
    )

    df_events_captured = df_patients_with_events.groupby("id").filter(lambda x: x["pred"].sum() > 0)

    return len(df_events_captured) / len(df_patients_with_events)

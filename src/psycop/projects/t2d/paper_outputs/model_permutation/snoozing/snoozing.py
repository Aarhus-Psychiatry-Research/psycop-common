import datetime as dt

import pandas as pd


def snoozing_filter(
    dates: pd.Series,
    predictions: pd.Series,
    snoozing_timedelta: dt.timedelta,
    snooze_on: set,
) -> tuple[list[dt.datetime], list[int]]:
    """
    Filter out all predictions that are within snoozing_threshold of each other.
    """
    # sort the dates and predictions by date
    date_pred = sorted(zip(dates, predictions), key=lambda x: x[0])
    snooze_until = None
    dates_filtered = []
    preds_filtered = []

    for date, pred in date_pred:
        if snooze_until is not None and (date < snooze_until):  # type: ignore
            continue  # type: ignore
        if pred in snooze_on:
            snooze_until = date + snoozing_timedelta

        dates_filtered.append(date)
        preds_filtered.append(pred)

    return dates_filtered, preds_filtered


def snooze_dataframe(
    df: pd.DataFrame,
    prediction_column_name: str = "prediction",
    time_column_name: str = "date",
    id_column_name: str = "id",
    snoozing_timedelta: dt.timedelta = dt.timedelta(days=90),  # noqa: B008
    snooze_on_value: int = 1,
) -> pd.DataFrame:
    """
    Filter out all predictions that are within snoozing_threshold of each other.
    """
    # use a group by to split the dataframe into individual dataframes
    # this is much faster than the above method
    snooze_on = {snooze_on_value}

    if len(set(df[prediction_column_name].unique())) != 2:
        raise ValueError("Predictions must be binary for snoozing")

    ids, f_dates, f_preds = [], [], []
    for ent_id, group in df.groupby(id_column_name):
        dates = group[time_column_name]
        predictions = group[prediction_column_name]

        filtered_dates, filtered_predictions = snoozing_filter(
            dates=dates,
            predictions=predictions,
            snoozing_timedelta=snoozing_timedelta,
            snooze_on=snooze_on,
        )
        ids.extend([ent_id] * len(filtered_dates))
        f_dates.extend(filtered_dates)
        f_preds.extend(filtered_predictions)

    return pd.DataFrame(
        {
            time_column_name: f_dates,
            prediction_column_name: f_preds,
            id_column_name: ids,
        },
    )

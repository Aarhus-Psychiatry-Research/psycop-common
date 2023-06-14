from pathlib import Path

import plotnine as pn
from care_ml.model_evaluation.config import (
    COLOURS,
    EVAL_RUN,
    FIGURES_PATH,
    PN_THEME,
)
from care_ml.model_evaluation.data.load_true_data import load_eval_df
from care_ml.utils.best_runs import Run


def incidence_by_time_until_outcome_pipeline(run: Run, path: Path):
    df = load_eval_df(
        wandb_group=run.group.name,
        wandb_run=run.name,
    )

    only_positives = df[df["y"] == 1]
    only_positives["time_from_pred_to_event"] = [
        (row["outcome_timestamps"] - row["pred_timestamps"]).total_seconds() / 60 / 60
        for idx, row in only_positives.iterrows()
    ]

    (
        pn.ggplot(only_positives, pn.aes(x="time_from_pred_to_event"))  # type: ignore
        + pn.geom_histogram(binwidth=8, fill=COLOURS["blue"], color="black")
        + pn.scale_x_reverse(
            breaks=range(
                0,
                int(only_positives["time_from_pred_to_event"].max() + 2),
                8,
            ),
        )
        + pn.labs(x="Hours from first positive label to event", y="Count")
        + PN_THEME
    ).save(
        path / "time_from_first_positive_to_event.png",
        dpi=300,
    )


if __name__ == "__main__":
    incidence_by_time_until_outcome_pipeline(EVAL_RUN, FIGURES_PATH)

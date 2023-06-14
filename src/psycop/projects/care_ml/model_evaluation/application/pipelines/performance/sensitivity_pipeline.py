from pathlib import Path

import pandas as pd
import plotnine as pn
from care_ml.model_evaluation.config import (
    COLOURS,
    EVAL_RUN,
    FIGURES_PATH,
    MODEL_NAME,
    PN_THEME,
    TEXT_EVAL_RUN,
    TEXT_FIGURES_PATH,
)
from care_ml.utils.best_runs import Run
from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df


def sensitivity_pipeline(run: Run, path: Path):
    eval_ds = run.get_eval_dataset(
        custom_columns=["outcome_coercion_type_within_2_days"],
    )

    output_df = get_auroc_by_input_df(
        eval_dataset=eval_ds,  # type: ignore
        input_values=eval_ds.custom_columns["outcome_coercion_type_within_2_days"],  # type: ignore
        input_name="outcome_type",
        bin_continuous_input=False,
        n_bootstraps=1000,
    )

    types = pd.Series(
        eval_ds.custom_columns["outcome_coercion_type_within_2_days"],  # type: ignore
    ).value_counts()

    output_df = output_df.merge(types, left_on="outcome_type", right_on=types.index)

    output_df["outcome_type"] = output_df["outcome_type"].replace(
        {
            3: "Mechanical restraint",
            2: "Chemical restraint",
            1: "Manual restraint",
        },
    )

    output_df = output_df[output_df["outcome_type"] != 0]

    output_df["proportion_in_bin"] = output_df[
        "outcome_coercion_type_within_2_days"
    ] / sum(output_df.outcome_coercion_type_within_2_days)
    output_df["outcome_coercion_type_within_2_days"] = output_df[
        "outcome_coercion_type_within_2_days"
    ].astype(int)

    (
        pn.ggplot(
            output_df,
            pn.aes("outcome_type", "sensitivity"),
        )
        + pn.geom_bar(
            pn.aes(x="outcome_type", y="proportion_in_bin"),
            stat="identity",
            position="identity",
            fill=COLOURS["blue"],
        )
        + pn.geom_text(
            pn.aes(y="proportion_in_bin", label="outcome_coercion_type_within_2_days"),
            va="bottom",
            size=11,
        )
        + pn.scale_x_discrete(
            limits=output_df.sort_values("outcome_coercion_type_within_2_days")[
                "outcome_type"
            ].to_list(),
        )
        + pn.geom_pointrange(
            pn.aes(ymin="ci_lower", ymax="ci_upper"),
            size=0.5,
        )
        + pn.labs(
            x="Outcome type",
            y="AUROC",
            title=f"{MODEL_NAME[run.name]} Performane by Outcome Type",
        )
        + PN_THEME
    ).save(
        path / "sensitivity_by_outcome_type.png",
    )


if __name__ == "__main__":
    sensitivity_pipeline(EVAL_RUN, FIGURES_PATH)
    sensitivity_pipeline(TEXT_EVAL_RUN, TEXT_FIGURES_PATH)

import polars as pl
from wasabi import Printer

from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline

msg = Printer(timestamp=True)

if __name__ == "__main__":
    pipeline_inputs = get_best_eval_pipeline().inputs

    flattened_dataset = pl.concat(
        [
            pipeline_inputs.get_flattened_split_as_lazyframe(split=split)  # type: ignore
            for split in ["train", "test", "val"]
        ],
        how="vertical",
    )

    outcome_columns = flattened_dataset.select(pl.col("^outc_.*$")).columns

    for col_name in outcome_columns:
        lookahead_distance = int(col_name.split("within_")[1].split("_days")[0])

        outcomes = flattened_dataset.select(pl.col(col_name)).collect()
        total = outcomes.shape[0]

        n_positive = outcomes.filter(pl.col(col_name) == 1).shape[0]
        percent_positive = f"{round(n_positive / total*100, 1)}%"

        n_negative = (
            outcomes.select(pl.col(col_name)).filter(pl.col(col_name) == 0).shape[0]
        )
        percent_negative = f"{round(n_negative / total*100, 1)}%"

        msg.info(
            f"{lookahead_distance}: pos: {n_positive:,} ({percent_positive}) | neg: {n_negative:,} ({percent_negative})",
        )

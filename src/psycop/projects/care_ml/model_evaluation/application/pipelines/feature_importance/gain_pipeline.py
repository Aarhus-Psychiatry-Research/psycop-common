"""Pipeline for creating figure and table with information gain"""

from typing import Literal

from care_ml.model_evaluation.config import (
    EVAL_RUN,
    FIGURES_PATH,
    TABLES_PATH,
    TEXT_EVAL_RUN,
    TEXT_FIGURES_PATH,
    TEXT_TABLES_PATH,
)
from care_ml.model_evaluation.data.load_true_data import (
    load_file_from_pkl,
)
from care_ml.model_evaluation.figures.feature_importance.gain_plot import (
    plot_gain,
)
from psycop.common.model_evaluation.feature_importance.feature_importance_table import (
    generate_feature_importances_table,
)


def gain_pipeline(model: Literal["baseline", "text"], top_n: int = 20):
    """Pipeline for running gain plot and gain table

    Args:
        model (Literal["baseline", "text"]): Which model to use.
        top_n (int, optional): How many features to include in gain plot. Defaults to 20
    """
    if model == "baseline":
        run, f_path, t_path = EVAL_RUN, FIGURES_PATH, TABLES_PATH
    elif model == "text":
        run, f_path, t_path = TEXT_EVAL_RUN, TEXT_FIGURES_PATH, TEXT_TABLES_PATH
    else:
        raise ValueError(f"model is {model}, but must be 'baseline' or 'text'")

    pipe = load_file_from_pkl(
        wandb_group=run.group.name,
        wandb_run=run.name,
        file_name="pipe.pkl",
    )

    feature_importance_dict = dict(
        zip(
            pipe.named_steps.model.feature_names,
            pipe.named_steps.model.feature_importances_,
        ),
    )

    # create and save gain plot
    plot_gain(
        feature_importance_dict=feature_importance_dict,
        save_path=f_path / "gain",
        top_n=top_n,
        model=model,
    )

    # create and save gain table
    gain_df = generate_feature_importances_table(
        feature_importance_dict=feature_importance_dict,
        output_format="df",
    )
    gain_df.to_csv(path_or_buf=t_path / "gain_table.csv")  # type: ignore


if __name__ == "__main__":
    gain_pipeline(model="baseline")
    gain_pipeline(model="text")

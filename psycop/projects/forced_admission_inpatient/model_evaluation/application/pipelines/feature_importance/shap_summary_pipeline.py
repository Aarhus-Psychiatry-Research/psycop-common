from typing import TYPE_CHECKING, Literal

from psycop.common.model_training.data_loader.utils import (
    load_and_filter_split_from_cfg,
)
from psycop.common.model_training.utils.col_name_inference import (
    infer_outcome_col_name,
    infer_predictor_col_name,
)
from psycop.projects.forced_admission_inpatient.model_evaluation.config import EVAL_RUN
from psycop.projects.forced_admission_inpatient.model_evaluation.data.load_true_data import (
    load_file_from_pkl,
    load_fullconfig,
)
from psycop.projects.forced_admission_inpatient.model_evaluation.figures.feature_importance.shap.get_shap_values import (
    generate_shap_values,
)
from psycop.projects.forced_admission_inpatient.model_evaluation.figures.feature_importance.shap.shap_plots import (
    plot_shap_summary,
)

if TYPE_CHECKING:
    from psycop.common.model_training.config_schemas.full_config import FullConfigSchema


def shap_summary_pipeline(model: Literal["baseline"], top_n: int = 20):
    """Pipeline for running gain plot and gain table

    Args:
        model (Literal["baseline"): Which model to use.
        top_n (int, optional): How many features to include in gain plot. Defaults to 20
    """
    if model == "baseline":
        run = EVAL_RUN
    else:
        raise ValueError(f"model is {model}. model must be 'baseline'")

    cfg: FullConfigSchema = load_fullconfig(
        wandb_group=run.group.name,
        wandb_run=run.name,
    )  # type: ignore

    pipe = load_file_from_pkl(
        wandb_group=run.group.name,
        wandb_run=run.name,
        file_name="pipe.pkl",
    )

    dataset = load_and_filter_split_from_cfg(
        data_cfg=cfg.data,
        pre_split_cfg=cfg.preprocessing.pre_split,
        split="val",
    )

    feature_cols = infer_predictor_col_name(dataset)

    outcome_cols = infer_outcome_col_name(dataset, prefix="outc_bool")

    shap_values = generate_shap_values(
        features=dataset[feature_cols],
        outcome=dataset[outcome_cols],
        pipeline=pipe,
    )

    plot_shap_summary(shap_values=shap_values, model=model, max_display=top_n)  # type: ignore


if __name__ == "__main__":
    shap_summary_pipeline(model="baseline")

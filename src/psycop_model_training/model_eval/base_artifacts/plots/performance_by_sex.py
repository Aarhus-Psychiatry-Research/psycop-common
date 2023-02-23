from src.psycop_model_training.model_eval.dataclasses import EvalDataset
import pandas as pd


def plot_auc_by_sex(
    eval_dataset: EvalDataset,
):
    """Plot auc by sex"""
    df = pd.DataFrame(
        {
            "is_female": eval_dataset.is_female,
            "y": eval_dataset.y,
            "y_hat": eval_dataset.y_hat_probs,
        }
    )

    pass

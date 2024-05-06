import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from psycop.common.model_training.data_loader.utils import (
    load_and_filter_split_from_cfg,
)
from psycop.projects.forced_admission_outpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_outpatient.utils.pipeline_objects import (
    ForcedAdmissionOutpatientPipelineRun,
)


def perform_pca(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:

    # Convert NAs to 0s
    df = df.fillna(0)

    # Drop columns that don't start with 'pred_'
    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    df_filtered = df[pred_cols]

    # Perform PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df_filtered)

    pca_df = pd.DataFrame(components, columns=["component_1", "component_2"])

    # appende pca_df to df
    df['component_1'] = pca_df['component_1'].to_numpy()
    df['component_2'] = pca_df['component_2'].to_numpy()

    return df


def pca_on_eval_splits(run: ForcedAdmissionOutpatientPipelineRun, keep_only_positive_outcome: bool = True) -> pd.DataFrame:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    eval_df = pd.DataFrame(
    {
        "prediction_time_uuid": eval_ds.pred_time_uuids,
        "true": eval_ds.y,
        "prob": eval_ds.y_hat_probs,
        "pred": eval_ds.get_predictions_for_positive_rate(run.paper_outputs.pos_rate)[0],
    }
)
    cfg = run.inputs.cfg

    # Load features
    if cfg.data.splits_for_evaluation is not None:
        cfg.data.splits_for_evaluation = ["val"]
        
    eval_feature_df = pd.concat(
        [
            load_and_filter_split_from_cfg(
                data_cfg=cfg.data,
                pre_split_cfg=cfg.preprocessing.pre_split,
                split=split,  # type: ignore
            )
            for split in cfg.data.splits_for_evaluation # type: ignore
        ],
        ignore_index=True,
    )
    
    df = pd.merge(eval_df, eval_feature_df, on="prediction_time_uuid")
    
    if keep_only_positive_outcome:
        df = df[df['true']==1]
        
    # Perform PCA
    return perform_pca(df)


if __name__ == "__main__":
    pca_on_eval_splits(run=get_best_eval_pipeline())

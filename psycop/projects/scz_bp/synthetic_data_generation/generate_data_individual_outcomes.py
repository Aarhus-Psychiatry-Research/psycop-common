"""Generate synthetic data for each of the individual outcomes"""
import numpy as np
import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from synthcity.plugins import Plugins  # type: ignore
from synthcity.plugins.core.dataloader import GenericDataLoader  # type: ignore
from xgboost import XGBClassifier

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.loggers.mlflow_logger import MLFlowLogger
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric
from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import populate_scz_bp_registry
from psycop.projects.scz_bp.synthetic_data_generation.DummyLogger import DummyLogger

if __name__ == "__main__":
    populate_baseline_registry()
    populate_scz_bp_registry()

    SAVE_DIR = OVARTACI_SHARED_DIR / "scz_bp" / "synthetic_data"
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    for individual_outcome in ["bp", "scz"]:
        if individual_outcome == "bp":
            continue
        print(f"Running {individual_outcome}")
        best_experiment = f"sczbp/{individual_outcome}_only"
        base_config = (
            MlflowClientWrapper()
            .get_best_run_from_experiment(
                experiment_name=best_experiment, metric="all_oof_BinaryAUROC"
            )
            .get_config()
        )
        outcome_col_name = base_config["trainer"]["outcome_col_name"]
        uuid_col_name = base_config["trainer"]["uuid_col_name"]

        # get training data
        training_data_cfg = {"data": base_config["trainer"]["training_data"]}
        training_data = BaselineRegistry.resolve(training_data_cfg)["data"].load()
        # get preprocessing pipeline
        preprocessing_pipeline_cfg = {"preproc": base_config["trainer"]["preprocessing_pipeline"]}
        preprocessing_pipeline = BaselineRegistry.resolve(preprocessing_pipeline_cfg)["preproc"]
        preprocessing_pipeline._logger = DummyLogger()
        # load and apply the preprocessing pipeline
        ## only train on positive examples
        training_data = training_data.filter(pl.col(outcome_col_name) == 1)
        training_data_preprocessed = preprocessing_pipeline.apply(data=training_data)

        # instantiate sklearn pipe with imputation and scaler
        pipe = Pipeline(
            [("imputer", SimpleImputer(strategy="most_frequent")), ("scaler", StandardScaler())]
        )
        y = training_data_preprocessed[outcome_col_name].to_list()
        training_data = training_data_preprocessed.drop(
            columns=[uuid_col_name, "dw_ek_borger", outcome_col_name]
        )
        training_data_cols = training_data.columns
        training_data_normalized = pipe.fit_transform(training_data)
        training_data_normalized = pd.DataFrame(
            training_data_normalized, columns=training_data_cols
        )

        # count number of positive cases
        n_positives = sum(y)
        # generate 2 times as many (sample multiplier = 2)
        n_to_generate = n_positives * 2

        # fit DDPM
        data_loader = GenericDataLoader(training_data_normalized)

        model_params = {"n_iter": 4000, "batch_size": 1000, "num_timesteps": 500.0, "lr": 0.0005}

        logger = MLFlowLogger(experiment_name="sczbp/ddpm_lr")

        model = Plugins().get("ddpm", **model_params)
        model.fit(data_loader)
        logger.log_metric(
            CalculatedMetric(name="loss", value=model.loss_history["loss"].values[-1])
        )

        generated_data = model.generate(count=n_to_generate).dataframe()

        # un-scale generated data
        unnormalised_generated_data = pipe.named_steps["scaler"].inverse_transform(generated_data)
        unnormalised_generated_data = pd.DataFrame(
            unnormalised_generated_data, columns=training_data_cols
        )
        # round ordinal values to ints
        ordinals = [col for col in unnormalised_generated_data.columns if "fallback_0" in col]
        unnormalised_generated_data[ordinals] = unnormalised_generated_data[ordinals].round(0)

        # calculate detection performance
        y_detection = np.concatenate([np.repeat([1], n_to_generate), np.repeat([0], n_positives)])
        score = cross_val_score(
            XGBClassifier(),
            X=pd.concat([unnormalised_generated_data, training_data]),
            y=y_detection,
            scoring="roc_auc",
        )

        logger.log_metric(CalculatedMetric(name="detection_auc_mean", value=score.mean()))
        for i, val in enumerate(score):
            logger.log_metric(CalculatedMetric(name=f"detection_auc_fold_{i}", value=val))

        # add metadata columns
        unnormalised_generated_data[outcome_col_name] = 1
        synthetic_id = [f"synthetic_{i}" for i in range(unnormalised_generated_data.shape[0])]
        unnormalised_generated_data[uuid_col_name] = synthetic_id
        unnormalised_generated_data["dw_ek_borger"] = synthetic_id

        save_name = f"{individual_outcome}_dppm_lr_0.0005_n_iter_{model_params['n_iter']}_n_timesteps_{model_params['num_timesteps']}_only_positive_{n_to_generate}.parquet"
        # save as parquet to disk
        unnormalised_generated_data.to_parquet(SAVE_DIR / save_name)

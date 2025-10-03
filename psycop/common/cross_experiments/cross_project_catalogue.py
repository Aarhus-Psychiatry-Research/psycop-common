from datetime import datetime
from typing import Literal

import pandas as pd
from psycop.common.cross_experiments.project_getters.ect_getter import ECTGetter
from psycop.common.cross_experiments.project_getters.restraint_getter import RestraintGetter
from psycop.common.cross_experiments.project_getters.cvd_getter import CVDGetter
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg

CROSS_EXPERIMENTS_BASE_PATH = "E:/shared_resources/cross_experiments/"


class ModelCatalogue:
    def __init__(self, projects: list[Literal["CVD", "ECT", "Restraint"]] = ["CVD", "ECT", "Restraint"]):
        self.projects = projects
        self.project_getters = {"CVD": CVDGetter, "ECT": ECTGetter, "Restraint": RestraintGetter}
        self.project_getters = {k: v for k, v in self.project_getters.items() if k in self.projects}

    def get_eval_dfs(self):
        return {k: v.get_eval_df() for k, v in self.project_getters.items()}

    def get_feature_set_dfs(self):
        return {k: v.get_feature_set_df() for k, v in self.project_getters.items()}

    def get_cfgs(self):
        return {k: v.get_cfg() for k, v in self.project_getters.items()}

    def retrain_and_test_from_configs(
        self,
        split_filter: Literal[
            "regional_data_filter", "outcomestratified_split_filter"
        ] = "outcomestratified_split_filter",
        experiment_name: str = "models_retrained_from_catalogue",
    ):
        cfgs = self.get_cfgs()

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = f"{experiment_name}_{date_str}"
        experiment_path = CROSS_EXPERIMENTS_BASE_PATH + experiment_name

        auc_rocs = {}

        for project, cfg in cfgs.items():
            print(f"Retraining model for project {project}")

            project_path = experiment_path + f"/{project}"

            # if imported cfg is set to geographic split, start by removing geographic split-specific args
            if cfg['trainer']['training_preprocessing_pipeline']['*']['split_filter']['@preprocessing'] == "regional_data_filter":
                cfg = (
                    cfg.rem("trainer.training_preprocessing_pipeline.*.split_filter.regional_move_df")
                    .rem(
                        "trainer.training_preprocessing_pipeline.*.split_filter.timestamp_col_name"
                    )
                    .rem("trainer.training_preprocessing_pipeline.*.split_filter.region_col_name")
                    .rem(
                        "trainer.training_preprocessing_pipeline.*.split_filter.timestamp_cutoff_col_name")
                    )
                
            # mutate config paths and filter
            cfg = (
                cfg.mut("logger.*.mlflow.experiment_name", experiment_name)
                .mut("logger.*.disk_logger.run_path", project_path)
                .mut("trainer.training_preprocessing_pipeline.*.split_filter.@preprocessing", split_filter)
            )

            # if desired split filter is geopgraphic, re-add necessary filters
            if split_filter == "regional_data_filter":
                cfg = (
                    cfg.add("trainer.training_preprocessing_pipeline.*.split_filter.regional_move_df", None)
                    .add(
                        "trainer.training_preprocessing_pipeline.*.split_filter.timestamp_col_name", "timestamp"
                    )
                    .add("trainer.training_preprocessing_pipeline.*.split_filter.region_col_name", "region")
                    .add(
                        "trainer.training_preprocessing_pipeline.*.split_filter.timestamp_cutoff_col_name",
                        "first_regional_move_timestamp",
                    )
                )

            auc_roc = train_baseline_model_from_cfg(cfg)

            auc_rocs[project] = auc_roc

        auc_rocs_df = pd.DataFrame.from_dict(auc_rocs, orient="index", columns=["auc_roc"])
        auc_rocs_df.to_csv(f"{experiment_path}/auc_rocs.csv")

        return auc_rocs


if __name__ == "__main__":
    model_catalogue = ModelCatalogue(projects=["CVD", "ECT", "Restraint"])
    auc_rocs = model_catalogue.retrain_and_test_from_configs()
    print(auc_rocs)

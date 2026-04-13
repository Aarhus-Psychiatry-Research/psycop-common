from datetime import datetime
from typing import Literal, Optional

import pandas as pd

from psycop.common.cross_experiments.project_getters.cvd_getter import CVDGetter
from psycop.common.cross_experiments.project_getters.ect_getter import ECTGetter
from psycop.common.cross_experiments.project_getters.fa_inpatient_getter import (
    ForcedAdmissionsInpatientGetter,
)
from psycop.common.cross_experiments.project_getters.restraint_getter import RestraintGetter
from psycop.common.cross_experiments.project_getters.scz_bp_getter import SczBpGetter
from psycop.common.cross_experiments.project_getters.t2d_getter import T2DGetter
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig

CROSS_EXPERIMENTS_BASE_PATH = "E:/shared_resources/cross_experiments/"


class ModelCatalogue:
    def __init__(
        self,
        projects: Optional[list[Literal["CVD", "ECT", "Restraint", "FAI", "SCZ_BP", "T2D"]]] = None,
    ):
        if projects is None:
            projects = ["CVD", "ECT", "Restraint", "FAI", "SCZ_BP", "T2D"]
        self.projects = projects
        self.project_getters = {
            "CVD": CVDGetter,
            "ECT": ECTGetter,
            "Restraint": RestraintGetter,
            "FAI": ForcedAdmissionsInpatientGetter,
            "SCZ_BP": SczBpGetter,
            "T2D": T2DGetter,
        }
        self.project_getters = {k: v for k, v in self.project_getters.items() if k in self.projects}

    def get_eval_dfs(self) -> dict[str, pd.DataFrame]:
        return {k: v.get_eval_df() for k, v in self.project_getters.items()}

    def get_feature_set_dfs(self) -> dict[str, pd.DataFrame]:
        return {k: v.get_feature_set_df() for k, v in self.project_getters.items()}

    def get_cfgs(self) -> dict[str, PsycopConfig]:
        return {k: v.get_cfg() for k, v in self.project_getters.items()}

    def get_predicted_positive_rates(self) -> dict[str, float]:
        return {k: v.predicted_positive_rate for k, v in self.project_getters.items()}

    def get_hyperparameter_tuning_cfgs(self) -> dict[str, PsycopConfig]:  # type: ignore
        return {
            k: v.get_hyperparameter_tuning_cfg()  # type: ignore
            for k, v in self.project_getters.items()
            if hasattr(v, "get_hyperparameter_tuning_cfg")
        }

    def retrain_and_test_from_configs_(
        self,
        project: str,
        cfg: PsycopConfig,
        experiment_name: str,
        split_filter: Literal["regional_data_filter", "outcomestratified_split_filter"],
    ) -> dict[str, str | float]:
        print(f"Retraining model for project {project}")

        project_path = CROSS_EXPERIMENTS_BASE_PATH + experiment_name + f"/{project}"

        # if imported cfg is set to geographic split, start by removing geographic split-specific args
        if (
            cfg["trainer"]["training_preprocessing_pipeline"]["*"]["split_filter"]["@preprocessing"]
            == "regional_data_filter"
        ):
            cfg = (
                cfg.rem("trainer.training_preprocessing_pipeline.*.split_filter.regional_move_df")
                .rem("trainer.training_preprocessing_pipeline.*.split_filter.timestamp_col_name")
                .rem("trainer.training_preprocessing_pipeline.*.split_filter.region_col_name")
                .rem(
                    "trainer.training_preprocessing_pipeline.*.split_filter.timestamp_cutoff_col_name"
                )
            )

        # mutate config paths and filter
        updated_cfg = PsycopConfig().from_str(f"""
        [logger]
        [logger.*]
        [logger.*.mlflow]
        experiment_name = {experiment_name}

        [logger.*.disk_logger]
        run_path = {project_path}

        [trainer]
        [trainer.training_preprocessing_pipeline]
        [trainer.training_preprocessing_pipeline.*]
        [trainer.training_preprocessing_pipeline.*.split_filter]
        @preprocessing = {split_filter}
        """)

        cfg = cfg.merge(updated_cfg)

        # if desired split filter is geopgraphic, re-add necessary filters
        if split_filter == "regional_data_filter":
            cfg = (
                cfg.add(
                    "trainer.training_preprocessing_pipeline.*.split_filter.regional_move_df", None
                )
                .add(
                    "trainer.training_preprocessing_pipeline.*.split_filter.timestamp_col_name",
                    "timestamp",
                )
                .add(
                    "trainer.training_preprocessing_pipeline.*.split_filter.region_col_name",
                    "region",
                )
                .add(
                    "trainer.training_preprocessing_pipeline.*.split_filter.timestamp_cutoff_col_name",
                    "first_regional_move_timestamp",
                )
            )

        auc_roc = train_baseline_model_from_cfg(cfg)

        return {"project": project, "auc": auc_roc}

    def retrain_and_test_from_configs(
        self,
        experiment_name: str = "models_retrained_from_catalogue",
        split_filter: Literal[
            "regional_data_filter", "outcomestratified_split_filter"
        ] = "outcomestratified_split_filter",
    ) -> pd.DataFrame:
        cfgs = self.get_cfgs()

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = f"{experiment_name}_{date_str}"

        auc_rocs = []

        for project, cfg in cfgs.items():
            auc_rocs.append(
                self.retrain_and_test_from_configs_(project, cfg, experiment_name, split_filter)
            )

        auc_rocs_df = pd.DataFrame(auc_rocs)
        auc_rocs_df.to_csv(f"{CROSS_EXPERIMENTS_BASE_PATH + experiment_name}/auc_rocs.csv")

        return auc_rocs_df


if __name__ == "__main__":
    model_catalogue = ModelCatalogue(projects=["Restraint"])
    auc_rocs = model_catalogue.retrain_and_test_from_configs()
    print(auc_rocs)

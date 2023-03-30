"""Code for generating a descriptive stats table."""
import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import wandb
from psycop_model_training.model_eval.dataclasses import EvalDataset

from psycop_model_evaluation.base_artifacts.tables.tables import output_table
from psycop_model_evaluation.utils.utils import bin_continuous_data


class DescriptiveStatsTable:
    """Class for generating a descriptive stats table."""

    def __init__(
        self,
        eval_dataset: EvalDataset,
    ) -> None:
        self.eval_dataset = eval_dataset

    def _get_column_header_df(self) -> pd.DataFrame:
        """Create empty dataframe with default columns headers.

        Returns:
            pd.DataFrame: Empty dataframe with default columns headers. Includes columns for category, two statistics and there units.
        """
        return pd.DataFrame(
            columns=["category", "stat_1", "stat_1_unit", "stat_2", "stat_2_unit"],
        )

    def _generate_age_stats(
        self,
    ) -> pd.DataFrame:
        """Add age stats to table 1."""

        df = self._get_column_header_df()

        age_mean = round(self.eval_dataset.age.mean(), 1)
        age_span = f"{self.eval_dataset.age.quantile(0.25)} - {self.eval_dataset.age.quantile(0.75)}"

        df = df.append(  # type: ignore
            {
                "category": "(visit_level) age (mean / 25-75 quartile interval)",
                "stat_1": age_mean,
                "stat_1_unit": "years",
                "stat_2": age_span,
                "stat_2_unit": "years",
            },
            ignore_index=True,
        )

        age_counts = bin_continuous_data(
            self.eval_dataset.age,  # type: ignore
            bins=[0, 17, *range(24, 75, 10)],
        )[0].value_counts()

        age_percentages = round(age_counts / len(self.eval_dataset.age) * 100, 1)  # type: ignore

        for i, _ in enumerate(age_counts):
            df = df.append(  # type: ignore
                {
                    "category": f"(visit level) age {age_counts.index[i]}",
                    "stat_1": int(age_counts.iloc[i]),
                    "stat_1_unit": "patients",
                    "stat_2": age_percentages.iloc[i],
                    "stat_2_unit": "%",
                },
                ignore_index=True,
            )

        return df

    def _generate_sex_stats(
        self,
    ) -> pd.DataFrame:
        """Add sex stats to table 1."""

        df = self._get_column_header_df()

        sex_counts = self.eval_dataset.is_female.value_counts()
        sex_percentages = sex_counts / len(self.eval_dataset.is_female) * 100  # type: ignore

        for i, n in enumerate(sex_counts):
            if n < 5:
                warnings.warn(
                    "WARNING: One of the sex categories has less than 5 individuals. This category will be excluded from the table.",
                )
                return df

            df = df.append(  # type: ignore
                {
                    "category": f"(visit level) {sex_counts.index[i]}",
                    "stat_1": int(sex_counts[i]),
                    "stat_1_unit": "patients",
                    "stat_2": sex_percentages[i],
                    "stat_2_unit": "%",
                },
                ignore_index=True,
            )

        return df

    def _generate_eval_col_stats(self) -> pd.DataFrame:
        """Generate stats for all eval_ columns to table 1.

        Finds all columns starting with 'eval_' and adds visit level
        stats for these columns. Checks if the column is binary or
        continuous and adds stats accordingly.
        """

        df = self._get_column_header_df()

        if (
            not hasattr(self.eval_dataset, "custom_columns")
            or self.eval_dataset.custom_columns is None
        ):
            return df

        eval_cols: list[dict[str, pd.Series]] = [
            {name: values}
            for name, values in self.eval_dataset.custom_columns.items()
            if name.startswith("eval_")
        ]

        for col in eval_cols:
            col_name = next(iter(col))
            col_values = col[col_name]

            if len(col_values.unique()) == 2:
                # Binary variable stats:
                col_count = col_values.value_counts()
                col_percentage = col_count / len(col_values) * 100

                if col_count[0] < 5 or col_count[1] < 5:
                    warnings.warn(
                        f"WARNING: One of categories in {col} has less than 5 individuals. This category will be excluded from the table.",
                    )
                else:
                    df = df.append(  # type: ignore
                        {
                            "category": f"(visit level) {col_name} ",
                            "stat_1": int(col_count[1]),
                            "stat_1_unit": "patients",
                            "stat_2": col_percentage[1],
                            "stat_2_unit": "%",
                        },
                        ignore_index=True,
                    )

            elif len(col_values.unique()) > 2:
                # Continuous variable stats:
                col_mean = round(col_values.mean(), 2)
                col_std = round(col_values.std(), 2)
                df = df.append(  # type: ignore
                    {
                        "category": f"(visit level) {col_name}",
                        "stat_1": col_mean,
                        "stat_1_unit": "mean",
                        "stat_2": col_std,
                        "stat_2_unit": "std",
                    },
                    ignore_index=True,
                )

            else:
                warnings.warn(
                    f"WARNING: {col_name} has only one value. This column will be excluded from the table.",
                )

        return df

    def _generate_visit_level_stats(
        self,
    ) -> pd.DataFrame:
        """Generate all visit level stats to table 1."""

        # Stats for eval_ cols
        df = self._generate_eval_col_stats()

        # General stats
        visits_followed_by_positive_outcome = self.eval_dataset.y.sum()

        visits_followed_by_positive_outcome_percentage = round(
            (visits_followed_by_positive_outcome / len(self.eval_dataset.ids) * 100),
            2,
        )

        df = df.append(  # type: ignore
            {
                "category": "(visit_level) visits followed by positive outcome",
                "stat_1": visits_followed_by_positive_outcome,
                "stat_1_unit": "visits",
                "stat_2": visits_followed_by_positive_outcome_percentage,
                "stat_2_unit": "%",
            },
            ignore_index=True,
        )

        return df

    def _calc_time_to_first_positive_outcome_stats(
        self,
        patients_with_positive_outcome_data: pd.DataFrame,
    ) -> tuple[float, float]:
        """Calculate mean time to first positive outcome (currently very
        slow)."""

        grouped_data = patients_with_positive_outcome_data.groupby("ids")

        time_to_first_positive_outcome = grouped_data.apply(
            lambda x: x["outcome_timestamps"].min() - x["pred_timestamps"].min(),  # type: ignore
        )

        # Convert to days (float)
        time_to_first_positive_outcome = (
            time_to_first_positive_outcome.dt.total_seconds() / (24 * 60 * 60)
        )  # Not using timedelta.days to keep higher temporal precision

        return round(time_to_first_positive_outcome.mean(), 2), round(
            time_to_first_positive_outcome.std(),
            2,
        )

    def _generate_patient_level_stats(
        self,
    ) -> pd.DataFrame:
        """Add patient level stats to table 1."""

        df = self._get_column_header_df()

        eval_df = pd.DataFrame(
            {
                "ids": self.eval_dataset.ids,
                "y": self.eval_dataset.y,
                "outcome_timestamps": self.eval_dataset.outcome_timestamps,
                "pred_timestamps": self.eval_dataset.pred_timestamps,
            },
        )

        # General stats
        patients_with_positive_outcome = eval_df[eval_df["y"] == 1]["ids"].unique()
        n_patients_with_positive_outcome = len(patients_with_positive_outcome)
        patients_with_positive_outcome_percentage = round(
            (n_patients_with_positive_outcome / len(eval_df["ids"].unique()) * 100),
            2,
        )

        df = df.append(  # type: ignore
            {
                "category": "(patient_level) patients_with_positive_outcome",
                "stat_1": n_patients_with_positive_outcome,
                "stat_1_unit": "visits",
                "stat_2": patients_with_positive_outcome_percentage,
                "stat_2_unit": "%",
            },
            ignore_index=True,
        )

        patients_with_positive_outcome_data = eval_df[
            eval_df["ids"].isin(patients_with_positive_outcome)
        ]

        (
            mean_time_to_first_positive_outcome,
            std_time_to_first_positive_outomce,
        ) = self._calc_time_to_first_positive_outcome_stats(
            patients_with_positive_outcome_data,
        )

        df = df.append(  # type: ignore
            {
                "category": "(patient level) time_to_first_positive_outcome",
                "stat_1": mean_time_to_first_positive_outcome,
                "stat_1_unit": "mean",
                "stat_2": std_time_to_first_positive_outomce,
                "stat_2_unit": "std",
            },
            ignore_index=True,
        )

        return df

    def generate_descriptive_stats_table(
        self,
        output_format: str = "df",
        save_path: Optional[Path] = None,
    ) -> Union[pd.DataFrame, wandb.Table]:
        """Generate descriptive stats table. Calculates relevant statistics
        from the evaluation dataset and returns a pandas dataframe or wandb
        table. If save_path is provided, the table is saved as a csv file.

        Args:
            output_format (str, optional): Output format. Defaults to "df".
            save_path (Optional[Path], optional): Path to save the table as a csv file. Defaults to None.

        Returns:
            Union[pd.DataFrame, wandb.Table]: Table 1.
        """
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.eval_dataset.age is not None:
            age_stats = self._generate_age_stats()

        if self.eval_dataset.is_female is not None:
            sex_stats = self._generate_sex_stats()

        visit_level_stats = self._generate_visit_level_stats()

        patient_level_stats = self._generate_patient_level_stats()

        table_1_df_list = [age_stats, sex_stats, visit_level_stats, patient_level_stats]
        table_1 = pd.concat(table_1_df_list, ignore_index=True)

        if save_path is not None:
            output_table(output_format="df", df=table_1)

            table_1.to_csv(save_path, index=False)

        return output_table(output_format=output_format, df=table_1)  # type: ignore

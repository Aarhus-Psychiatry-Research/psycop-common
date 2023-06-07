"""Main feature generation."""
# %%
import logging

from psycop.projects.t2d.t2d_config import T2D_PROJECT_INFO

log = logging.getLogger()

# %%
from psycop.projects.t2d.feature_generation.specify_features import FeatureSpecifier

feature_specs = FeatureSpecifier(
    project_info=T2D_PROJECT_INFO,
    min_set_for_debug=False,  # Remember to set to False when generating full dataset
).get_feature_specs()

selected_specs = [spec for spec in feature_specs if "pred" in spec.get_col_str()]

# %%
# %reload_ext autoreload
# %autoreload 2

# %%
from psycop.common.feature_generation.data_checks.flattened.feature_describer import (
    generate_feature_description_df,
)
from psycop.projects.t2d.paper_outputs.selected_runs import BEST_EVAL_PIPELINE

out_dir = BEST_EVAL_PIPELINE.paper_outputs.paths.tables / "feature_description"
out_dir.mkdir(parents=True, exist_ok=True)

df = generate_feature_description_df(
    df=BEST_EVAL_PIPELINE.inputs.get_flattened_split_as_pd(split="train"),
    predictor_specs=selected_specs,  # type: ignore
)

# %%
import pandas as pd


def prettify_feature_description_df(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    df = input_df

    df = df.rename(
        {
            "N unique": "Unique values in predictor",
            "predictor df": "Predictor",
            "25.0-percentile": "25-percentile",
            "50.0-percentile": "50-percentile",
            "75.0-percentile": "75-percentile",
            "99-percentile": "75-percentile",
        },
        axis=1,
    )

    df["Predictor df"] = df["Predictor df"].replace(
        {
            "f0_disorders": "f0 - Organic disorders",
            "f1_disorders": "f1 - Substance abuse",
            "f2_disorders": "f2 - Psychotic disorders",
            "f3_disorders": "f3 - Mood disorders",
            "f4_disorders": "f4 - Neurotic & stress-related",
            "f5_disorders": "f5 - Eating & sleeping disorders",
            "f6_disorders": "f6 - Perosnality disorders",
            "f7_disorders": "f7 - Mental retardation",
            "f8_disorders": "f8 - Developmental disorders",
            "hyperkinetic_disorders": "f9 - Child & adolescent disorders",
        },
    )

    df = df.drop(
        ["1.0-percentile", "99.0-percentile", "Histogram", "Proportion missing"],
        axis=1,
    )

    df = df.sort_values("Predictor df", ascending=True)

    return df


prettified = prettify_feature_description_df(input_df=df)
print(prettified)

# %%
from psycop.projects.t2d.paper_outputs.selected_runs import BEST_EVAL_PIPELINE

predictor_description_path = (
    BEST_EVAL_PIPELINE.paper_outputs.paths.tables / "predictor_description.csv"
)

prettified.to_csv(predictor_description_path)

# %%
from psycop.projects.t2d.paper_outputs.aggregate_eval.md_objects import MarkdownTable

md = MarkdownTable(
    title="## **eTable 2**: Descriptive statistics for predictors (a list of abbreviations is inserted below the table)",
    file_path=predictor_description_path,
    description="""**ALAT**: Alanine aminotransferase
**p-Glc**: Plasma glucose.
**BMI**: Body mass index.
**CRP**: C-reactive protein.
**EGFR**: Estimated glomerular filtration rate.
**LDL**: Low-density lipoprotein.
**GERD**: Gastro-esophageal reflux disorder.
**HbA1c**: Hemoglobin A1c.
**HDL**: High-density lipoprotein.
**OGTT**: Oral glucose-tolerance test.
**Glc**: Glucose.
**NASSA**: Noradrenergic and specific serotonergic antidepressants.
**SNRI**: Serotonin and noradrenaline reuptake inhibitor.
**SSRI**: Selective serotonin reuptake inhibitor.
**TCA**: Tricyclic antidepressants.
""",
).get_markdown()

with (BEST_EVAL_PIPELINE.paper_outputs.paths.tables / "predictor_description.md").open(
    "+w",
) as f:
    f.write(md)

# %%

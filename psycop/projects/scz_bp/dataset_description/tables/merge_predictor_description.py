import polars as pl

from psycop.common.feature_generation.data_checks.utils import save_df_to_pretty_html_table
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR

left = pl.read_csv("feature_description_left.csv", separator=";")
right = pl.read_csv("feature_description_right.csv", separator=";")


tab = pl.concat([left, right], how="horizontal")

tab = tab.with_columns(
    pl.when(pl.col("Feature name").str.starts_with("tfidf"))
    .then(pl.lit("TF-IDF 1000"))
    .otherwise(pl.lit("Structured"))
    .alias("Source"),
    pl.col("Feature name").str.replace("tfidf-1000_", ""),
    pl.col("English name").str.replace("tfidf-1000_", ""),
).select(
    "Source",
    "Feature name",
    "English name",
    "Lookbehind period",
    "Resolve multiple",
    "Fallback strategy",
    "Static",
    "Mean",
    "N. unique",
    "Proportion using fallback",
)

tab.to_pandas().to_excel(
    SCZ_BP_EVAL_OUTPUT_DIR / "feature_description_with_translation.xlsx", index=False
)

save_df_to_pretty_html_table(
    df=tab.to_pandas(),
    path=SCZ_BP_EVAL_OUTPUT_DIR / "feature_description_with_translation.html",
    subtitle="eTable 1: Predictor description. “Source” refers to whether the feature is based on structured data or term frequency – inverse document frequency (TF-IDF) 1000. “Feature name” is the name of the structured feature/the token from TF-IDF. English name is a translation of the Danish token to English. “Lookbehind period” refers to how far back in time from the prediction time to look for values. “Resolve multiple” indicates the aggregation function, i.e. how to handle multiple values in the lookbehind. “Fallback strategy” indicates which value set if no values exist in the lookbehind for the predictor. “Static” indicates whether the value can change over time or is static (e.g. sex). “Mean” indicates the mean value. “N. unique” represents the number of unique values. “Proportion” using fallback refers to how large a proportion of the values in the column consists of the fallback value.",  # noqa: RUF001
)

import pandas as pd
import plotnine as pn

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.first_clozapine_prescription import (
    get_first_clozapine_prescription,
)

view = "[FOR_labka_alle_blodprover_inkl_2021_feb2022]"
sql = "SELECT * FROM [fct]." + view + " WHERE npukode = 'NPU04114'"

clozapine_lab_results = sql_load(sql)


df_first_clozapine_lab_results = (
    clozapine_lab_results.sort_values("datotid_sidstesvar", ascending=True)
    .rename(columns={"datotid_sidstesvar": "timestamp"})
    .groupby("dw_ek_borger")
    .first()
    .reset_index(drop=False)
)


min_date = pd.Timestamp("2016-10-01")

df_prescription = get_first_clozapine_prescription()

min_date = pd.Timestamp("2016-10-01")

df_prescription_stable = df_prescription[df_prescription["timestamp"] > min_date]
df_lab_results_stable = df_first_clozapine_lab_results[
    df_first_clozapine_lab_results["timestamp"] > min_date
]

merged_df_prescription_lab_results = pd.merge(
    df_lab_results_stable,
    df_prescription_stable,
    on="dw_ek_borger",
    suffixes=("_labresults", "_prescription"),
)

merged_df_prescription_lab_results["days_difference"] = (
    merged_df_prescription_lab_results["timestamp_labresults"]
    - merged_df_prescription_lab_results["timestamp_prescription"]
).dt.days

merged_df_prescription_lab_results["absolute_days_difference"] = (
    merged_df_prescription_lab_results["timestamp_labresults"]
    - merged_df_prescription_lab_results["timestamp_prescription"]
).dt.days.abs()

# Calculate the mean and median differences
mean_diff = merged_df_prescription_lab_results["absolute_days_difference"].mean()
median_diff = merged_df_prescription_lab_results["days_difference"].median()

print(
    "Differences between lab results and prescription on text outcome timestamps ( stable data 2016-2021)"
)
print(f"Mean difference in days: {mean_diff}")
print(f"Median difference in days: {median_diff}")

# Plot the histogram of the differences
hist_plot = (
    pn.ggplot(merged_df_prescription_lab_results, pn.aes(x="days_difference"))
    + pn.geom_histogram(binwidth=1, fill="blue", color="blue", boundary=0)
    + pn.scale_x_continuous(limits=(-1000, 1000))
    + pn.labs(
        title="Histogram of Timestamp Differences between lab and prescription in  2016-2021",
        x="Difference in Days",
        y="Frequency",
    )
)

print(hist_plot)


within_10_days = merged_df_prescription_lab_results["days_difference"].between(-10, 10)
num_within_10_days = within_10_days.sum()

# Determine the number of dw_ek_borger entries greater than 10 days
greater_than_10_days = ~within_10_days
num_greater_than_10_days = greater_than_10_days.sum()

# Calculate the total number of dw_ek_borger entries
total_num_dw_ek_borger = merged_df_prescription_lab_results["dw_ek_borger"].nunique()

# Calculate percentages
percent_within_10_days = (num_within_10_days / total_num_dw_ek_borger) * 100
percent_greater_than_10_days = (num_greater_than_10_days / total_num_dw_ek_borger) * 100

print(
    "Analysis on cut-off for all cases within +-10 days for lab results and prescription (2016-2021)"
)
print(f"Total number of dw_ek_borger after merged_df : {total_num_dw_ek_borger}")
print(f"Number of timestamps within -10 to 10 days: {num_within_10_days}")
print(f"Number of timestamps greater than 10 days: {num_greater_than_10_days}")

print(f"Percentage of dw_ek_borger within -10 to 10 days: {percent_within_10_days:.2f}%")
print(f"Percentage of dw_ek_borger greater than 10 days: {percent_greater_than_10_days:.2f}%")

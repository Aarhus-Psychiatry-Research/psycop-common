import pandas as pd
import plotnine as pn

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.combine_text_structured_clozapine_outcome import (
    combine_structured_and_text_outcome,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.first_clozapine_prescription import (
    get_first_clozapine_prescription,
)

df_combined = combine_structured_and_text_outcome()


plot_all = (
    pn.ggplot(df_combined, pn.aes(x="timestamp"))
    + pn.geom_histogram(binwidth=30, fill="blue", color="blue")
    + pn.labs(
        title="Histogram of combined validated outcome timestamps 2013-2021 ",
        x="Dates",
        y="Frequency",
    )
)
print(plot_all)

df_combined["year"] = df_combined["timestamp"].dt.year

min_date = pd.Timestamp("2014-01-01")

df_combined_incident = df_combined[df_combined["timestamp"] > min_date]

plot_incident = (
    pn.ggplot(df_combined_incident, pn.aes(x="timestamp"))
    + pn.geom_histogram(binwidth=30, fill="red", color="green")
    + pn.labs(title="Histogram of incident outcome timestamps 2014-2021 ", x="Dates", y="Frequency")
)

print(plot_incident)
print("Number of unique outcomes in incident df 2014-2021 :", len(df_combined_incident))

cases_per_year = df_combined_incident.groupby("year").size()

# Calculate the mean number of cases per year
mean_cases_per_year = cases_per_year.mean()

print(f"Mean number of unique cases per year 2014-2021: {mean_cases_per_year}")


### prescription  validation ##
df_only_structured_prescriptions = get_first_clozapine_prescription()

min_date = pd.Timestamp("2016-10-01")

df_prescription_stable = df_only_structured_prescriptions[
    df_only_structured_prescriptions["timestamp"] > min_date
]
total_num_df_prescription_stable = df_prescription_stable["dw_ek_borger"].nunique()
print(
    f"total number of unique cases in prescription data 2016-2021:{total_num_df_prescription_stable}"
)

df_val_incident_stable = df_combined[df_combined["timestamp"] > min_date]
total_num_df_val_incident_stable = df_val_incident_stable["dw_ek_borger"].nunique()
print(f"total number of unique cases in text data 2016-2021:{total_num_df_val_incident_stable}")


merged_df_stable = pd.merge(
    df_val_incident_stable,
    df_prescription_stable,
    on="dw_ek_borger",
    suffixes=("_text", "_prescription"),
)


merged_df_stable["days_difference"] = (
    merged_df_stable["timestamp_text"] - merged_df_stable["timestamp_prescription"]
).dt.days

merged_df_stable["absolute_days_difference"] = (
    merged_df_stable["timestamp_text"] - merged_df_stable["timestamp_prescription"]
).dt.days.abs()

# Calculate the mean and median differences
mean_diff = merged_df_stable["absolute_days_difference"].mean()
median_diff = merged_df_stable["days_difference"].median()

print(
    "Differences between text validation and prescription on text outcome timestamps ( stable data 2016-2021)"
)
print(f"Mean difference in days: {mean_diff}")
print(f"Median difference in days: {median_diff}")

# Plot the histogram of the differences
hist_plot = (
    pn.ggplot(merged_df_stable, pn.aes(x="days_difference"))
    + pn.geom_histogram(binwidth=1, fill="blue", color="blue", boundary=0)
    + pn.scale_x_continuous(limits=(-1000, 1000))
    + pn.labs(
        title="Histogram of Timestamp Differences 2016-2021", x="Difference in Days", y="Frequency"
    )
)

print(hist_plot)


within_10_days = merged_df_stable["days_difference"].between(-10, 10)
num_within_10_days = within_10_days.sum()

# Determine the number of dw_ek_borger entries greater than 10 days
greater_than_10_days = ~within_10_days
num_greater_than_10_days = greater_than_10_days.sum()

# Calculate the total number of dw_ek_borger entries
total_num_dw_ek_borger = merged_df_stable["dw_ek_borger"].nunique()

# Calculate percentages
percent_within_10_days = (num_within_10_days / total_num_dw_ek_borger) * 100
percent_greater_than_10_days = (num_greater_than_10_days / total_num_dw_ek_borger) * 100

print("Analysis on cut-off for all cases within +-10 days (2016-2021)")
print(f"Total number of dw_ek_borger after merged_df : {total_num_dw_ek_borger}")
print(f"Number of timestamps within -10 to 10 days: {num_within_10_days}")
print(f"Number of timestamps greater than 10 days: {num_greater_than_10_days}")

print(f"Percentage of dw_ek_borger within -10 to 10 days: {percent_within_10_days:.2f}%")
print(f"Percentage of dw_ek_borger greater than 10 days: {percent_greater_than_10_days:.2f}%")

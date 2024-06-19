import pandas as pd
import plotnine as pn

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.first_clozapine_prescription import (
    get_first_clozapine_prescription,
)

view = "[raw_text_df_clozapine_outcome]"
sql = "SELECT * FROM [fct]." + view

raw_text_df = sql_load(sql)

number_of_rows = len(raw_text_df)

print("Number of rows in raw_text_df: ", len(raw_text_df))
print("Number of unique dw_ek_borger in raw_text_df: ", raw_text_df["dw_ek_borger"].nunique())


validated_path = (
    "E:/shared_resources/clozapine/text_outcome/validated_text_outcome_clozapine_v129.parquet"
)

df_val_all = pd.read_parquet(path=validated_path)

plot_all = (
    pn.ggplot(df_val_all, pn.aes(x="timestamp"))
    + pn.geom_histogram(binwidth=30, fill="blue", color="blue")
    + pn.labs(
        title="Histogram of text validated outcome timestamps 2013-2021 ", x="Dates", y="Frequency"
    )
)

print(plot_all)
print("Unique number of text validated outcomes 2013-2021 :", len(df_val_all))

df_val_all["year"] = df_val_all["timestamp"].dt.year

# Group by year and count the number of cases per year
cases_per_year = df_val_all.groupby("year").size()

# Calculate the mean number of cases per year
mean_cases_per_year = cases_per_year.mean()

print(f"Mean number of unique cases per year 2013-2021: {mean_cases_per_year}")

min_date = pd.Timestamp("2014-01-01")

df_val_incident = df_val_all[df_val_all["timestamp"] > min_date]

plot_incident = (
    pn.ggplot(df_val_incident, pn.aes(x="timestamp"))
    + pn.geom_histogram(binwidth=30, fill="red", color="green")
    + pn.labs(title="Histogram of incident outcome timestamps 2014-2021 ", x="Dates", y="Frequency")
)

print(plot_incident)
print("Number of unique outcomes in incident df 2014-2021 :", len(df_val_incident))

cases_per_year = df_val_incident.groupby("year").size()

# Calculate the mean number of cases per year
mean_cases_per_year = cases_per_year.mean()

print(f"Mean number of unique cases per year 2014-2021: {mean_cases_per_year}")


### prescription  validation ##
df_prescription = get_first_clozapine_prescription()

min_date = pd.Timestamp("2016-10-01")

df_prescription_stable = df_prescription[df_prescription["timestamp"] > min_date]
total_num_df_prescription_stable = df_prescription_stable["dw_ek_borger"].nunique()
print(
    f"total number of unique cases in prescription data 2016-2021:{total_num_df_prescription_stable}"
)

df_val_incident_stable = df_val_incident[df_val_incident["timestamp"] > min_date]
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

######## on non-stable 2013-2021 ############

merged_df = pd.merge(
    df_val_incident, df_prescription, on="dw_ek_borger", suffixes=("_text", "_prescription")
)

diff_dw_ek_borger = set(merged_df["dw_ek_borger"]).difference(set(df_val_incident["dw_ek_borger"]))


merged_df["days_difference"] = (
    merged_df["timestamp_text"] - merged_df["timestamp_prescription"]
).dt.days

merged_df["absolute_days_difference"] = (
    merged_df["timestamp_text"] - merged_df["timestamp_prescription"]
).dt.days.abs()

# Calculate the mean and median differences
mean_diff = merged_df["absolute_days_difference"].mean()
median_diff = merged_df["days_difference"].median()

print(
    "Differences between text validation and prescription on text outcome timestamps ( stable data 2016-2021)"
)

print(f"Mean difference in days: {mean_diff}")
print(f"Median difference in days: {median_diff}")

# Plot the histogram of the differences
hist_plot = (
    pn.ggplot(merged_df, pn.aes(x="days_difference"))
    + pn.geom_histogram(binwidth=1, fill="blue", color="blue", boundary=0)
    + pn.scale_x_continuous(limits=(-1000, 1000))
    + pn.labs(
        title="Histogram of non-stable Timestamp Differences", x="Difference in Days", y="Frequency"
    )
)

print(hist_plot)

within_10_days = merged_df["days_difference"].between(-10, 10)
num_within_10_days = within_10_days.sum()

# Determine the number of dw_ek_borger entries greater than 10 days
greater_than_10_days = ~within_10_days
num_greater_than_10_days = greater_than_10_days.sum()

# Calculate the total number of dw_ek_borger entries
total_num_dw_ek_borger = merged_df["dw_ek_borger"].nunique()

# Calculate percentages
percent_within_10_days = (num_within_10_days / total_num_dw_ek_borger) * 100
percent_greater_than_10_days = (num_greater_than_10_days / total_num_dw_ek_borger) * 100

print("Analysis on cut-off for all cases within +-10 days (2013-2021)")
print(f"Number of timestamps within -10 to 10 days: {num_within_10_days}")
print(f"Number of timestamps greater than 10 days: {num_greater_than_10_days}")
print(f"Total number of timestamps: {total_num_dw_ek_borger}")
print(f"Percentage of timestamps within -10 to 10 days: {percent_within_10_days:.2f}%")
print(f"Percentage of timestamps greater than 10 days: {percent_greater_than_10_days:.2f}%")

## unsure
unsure_path = "E:/shared_resources/clozapine/text_outcome/unsure_text_outcome_clozapine_v97.parquet"

df_unsure = pd.read_parquet(path=unsure_path)

print("")
print("UNSURE ROWS IN TEXT VALIDATION PROCESS")


print("number of unique dw_ek_borger in unsure_text_outcome")
print(len(df_unsure["dw_ek_borger"].unique()))

plot_unsure = (
    pn.ggplot(df_unsure, pn.aes(x="timestamp"))
    + pn.geom_histogram(binwidth=30, fill="red", color="yellow")
    + pn.labs(title="Histogram of unsure outcome timestamps ", x="Dates", y="Frequency")
)

print(plot_unsure)

print("number of rows in unsure")
print(len(df_unsure))

df_unsure["year"] = df_unsure["timestamp"].dt.year

rows_per_year = df_unsure.groupby("year").size()

# Calculate the mean number of cases per year
mean_rows_per_year = rows_per_year.mean()


print(f"Mean number of rows per year: {mean_cases_per_year}")

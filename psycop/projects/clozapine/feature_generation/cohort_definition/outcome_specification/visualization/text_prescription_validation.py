import pandas as pd
import plotnine as pn

from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.first_clozapine_prescription import (
    get_first_clozapine_prescription,
)

df_prescription = get_first_clozapine_prescription()

df_prescription["year"] = df_prescription["timestamp"].dt.year


plot_prescription = (
    pn.ggplot(df_prescription, pn.aes(x="timestamp"))
    + pn.geom_histogram(binwidth=30, fill="red", color="green")
    + pn.labs(title="Histogram of prescription outcome timestamps ", x="Dates", y="Frequency")
)

print(plot_prescription)
print("Number of outcomes in presciption data 2011-2021 :", len(df_prescription))

cases_per_year = df_prescription.groupby("year").size()

# Calculate the mean number of cases per year
mean_cases_per_year = cases_per_year.mean()

print(f"Mean number of cases per year: {mean_cases_per_year}")

min_date = pd.Timestamp("2016-10-01")

df_prescription_stable = df_prescription[df_prescription["timestamp"] > min_date]

plot_prescription_stable = (
    pn.ggplot(df_prescription_stable, pn.aes(x="timestamp"))
    + pn.geom_histogram(binwidth=30, fill="red", color="green")
    + pn.labs(title="Histogram of prescription outcome timestamps ", x="Dates", y="Frequency")
)

print(plot_prescription_stable)
print("Number of outcomes in presciption data 2016-2021 :", len(df_prescription_stable))

cases_per_year = df_prescription_stable.groupby("year").size()

# Calculate the mean number of cases per year
mean_cases_per_year = cases_per_year.mean()

print(f"Mean number of cases per year: {mean_cases_per_year}")

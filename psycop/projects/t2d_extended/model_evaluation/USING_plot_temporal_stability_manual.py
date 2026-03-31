"""OBS: Temporary solution. Only to be used while I cannot connect to the database."""

import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.array([2018, 2019, 2020, 2021, 2022, 2023])  # Convert years to numerical values for modeling
y = np.array([0.84, 0.83, 0.87, 0.80, 0.84, 0.80])

# Fit a linear model to the data
slope, intercept = np.polyfit(x, y, 1)  # Degree 1 for linear fit
y_fit = slope * x + intercept  # Linear model values

# Set a larger figure size for readability in publications
plt.figure(figsize=(8, 5), dpi=300)

# Plot data points
plt.plot(x, y, marker="o", linestyle="None", color="black", markersize=8)

# Color area
plt.fill_between(x, y, color="black", alpha=0.1)

# Plot linear model line
plt.plot(x, y_fit, linestyle="-", color="cornflowerblue", linewidth=1.5)

# Add axis labels and grid
plt.xlabel("Year", fontsize=12)
plt.ylabel("AUROC", fontsize=12)
plt.ylim(0.7, 1)
plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)

# Increase tick parameters for both x and y axes
plt.xticks(
    x, labels=["2018", "2019", "2020", "2021", "2022", "2023"], fontsize=10
)  # Adjust ticks to be years
plt.yticks(fontsize=10)

# Add legend with label
plt.legend(fontsize=10, frameon=False)

# Display the plot
plt.show()

# måske lav en blok med farve for hver anden år, så det er mere tydeligt, at det er tid
# farv areal under punkterne (som bjergkæde)
# ekstrapolér til fremtiden???


import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.array([2018, 2019, 2020, 2021, 2022, 2023])  # Convert years to numerical values for modeling
y = np.array([0.84, 0.83, 0.87, 0.80, 0.84, 0.80])

# Number of bootstrap samples
n_bootstraps = 1000
bootstrap_slopes = []

# Perform bootstrapping
for _ in range(n_bootstraps):
    indices = np.random.choice(len(x), len(x), replace=True)
    x_sample = x[indices]
    y_sample = y[indices]
    slope, intercept = np.polyfit(x_sample, y_sample, 1)
    bootstrap_slopes.append(slope)

# Calculate the 95% confidence interval for the slope
conf_interval = np.percentile(bootstrap_slopes, [2.5, 97.5])

# Check if the confidence interval covers 0
covers_zero = conf_interval[0] <= 0 <= conf_interval[1]

# Fit a linear model to the original data
slope, intercept = np.polyfit(x, y, 1)
y_fit = slope * x + intercept

# Set a larger figure size for readability in publications
plt.figure(figsize=(8, 5), dpi=300)

# Define hex colors for each year
#hex_colors = ["#ed220d", "#feae00", "#ff42a1", "#f27200", "#970e53", "#970e53"]
hex_colors = ["#093288", "#093288", "#093288", "#093288", "#093288", "#093288"]

# Plot data points with different hex colors
for i in range(len(x)):
    plt.plot(x[i], y[i], marker="o", linestyle="None", color=hex_colors[i], markersize=8)

# Plot linear model line
plt.plot(x, y_fit, linestyle="-", color="black", linewidth=1.5)

# Plot confidence interval
plt.fill_between(x, y_fit - conf_interval[1], y_fit + conf_interval[1], color="black", alpha=0.1)

# Add axis labels and grid
plt.xlabel("Year", fontsize=12)
plt.ylabel("AUROC", fontsize=12)
plt.ylim(0.7, 1)
plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)

# Increase tick parameters for both x and y axes
plt.xticks(
    x, labels=["2018", "2019", "2020", "2021", "2022", "2023"], fontsize=10
)  # Adjust ticks to be years
plt.yticks(fontsize=10)

# Add legend with label
plt.legend(fontsize=10, frameon=False)

# Display the plot
plt.show()

# Print if the confidence interval covers 0
print(f"Confidence interval for the slope: {conf_interval}")
print(f"Does the confidence interval cover 0? {'Yes' if covers_zero else 'No'}")





import statsmodels.api as sm


# Add intercept column
X = sm.add_constant(x)

# Fit linear regression
model = sm.OLS(y, X).fit()

# Extract slope and its standard error
slope = model.params[1]
slope_se = model.bse[1]

print("Slope:", slope)
print("Standard error:", slope_se)





import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Data
x = np.array([2018, 2019, 2020, 2021, 2022, 2023])
y = np.array([0.84, 0.83, 0.87, 0.80, 0.84, 0.80])

# ----- Linear model with statsmodels -----
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

slope = model.params[1]
slope_se = model.bse[1]

# Predictions + 95% CI for the regression line
pred = model.get_prediction(X)
pred_ci = pred.conf_int()
pred_mean = pred.predicted_mean

# ----- Matplotlib styling for clinical publication -----
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["font.size"] = 11

plt.figure(figsize=(7, 4), dpi=300)

# ----- Scatter points (hollow circles for grayscale print) -----
plt.scatter(
    x, y,
    s=60,
    facecolors="white",
    edgecolors="black",
    linewidth=1.2
)

# ----- Regression line -----
plt.plot(
    x, pred_mean,
    color="black",
    linewidth=1.8
)

# ----- Confidence interval -----
plt.fill_between(
    x,
    pred_ci[:, 0],
    pred_ci[:, 1],
    color="gray",
    alpha=0.18
)

# ----- Axis labels -----
plt.xlabel("Year", fontsize=12)
plt.ylabel("AUROC", fontsize=12)

# ----- Axis limits -----
plt.ylim(0.78, 0.89)
plt.xlim(2017.7, 2023.3)

# ----- Subtle grid -----
plt.grid(
    True,
    linestyle=":",
    linewidth=0.5,
    alpha=0.6
)

plt.tight_layout()


#plt.savefig("t2d-extended_linear-regression.png")

plt.show()

# Print numeric estimates
print("Slope:", slope)
print("Standard error:", slope_se)




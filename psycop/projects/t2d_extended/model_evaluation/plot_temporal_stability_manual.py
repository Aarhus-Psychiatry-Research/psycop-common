"""OBS: Temporary solution. Only to be used while I cannot connect to the database."""

import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.array([2018, 2019, 2020, 2021, 2022])  # Convert years to numerical values for modeling
y = np.array([0.84, 0.83, 0.87, 0.80, 0.82])

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
    x, labels=["2018", "2019", "2020", "2021", "2022"], fontsize=10
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
x = np.array([2018, 2019, 2020, 2021, 2022])  # Convert years to numerical values for modeling
y = np.array([0.84, 0.83, 0.87, 0.80, 0.82])

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
hex_colors = ["#ed220d", "#feae00", "#ff42a1", "#f27200", "#970e53"]

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
    x, labels=["2018", "2019", "2020", "2021", "2022"], fontsize=10
)  # Adjust ticks to be years
plt.yticks(fontsize=10)

# Add legend with label
plt.legend(fontsize=10, frameon=False)

# Display the plot
plt.show()

# Print if the confidence interval covers 0
print(f"Confidence interval for the slope: {conf_interval}")
print(f"Does the confidence interval cover 0? {'Yes' if covers_zero else 'No'}")

aurocs_training_end_date_2018 = {18: 0.8376074504993402, 19: 0.8331785757189495, 20: 0.8716330715094136, 21: 0.7956406780790461, 22: 0.8399708928832673, 23: 0.7985506465746504}
aurocs_training_end_date_2019 = {19: 0.8364743910231012, 20: 0.8613122432405681, 21: 0.7872254209668531, 22: 0.8276700817938417, 23: 0.7896403901045189}
aurocs_training_end_date_2020 = {20: 0.8670152208419689, 21: 0.7928056922527461, 22: 0.8338902756140362, 23: 0.7923104728996585}
aurocs_training_end_date_2021 = {21: 0.8009282114806378, 22: 0.8406449489498253, 23: 0.7943600358218161}
aurocs_training_end_date_2022 = {22: 0.8490573397279086, 23: 0.7975422369165425}
aurocs_training_end_date_2023 = {23: 0.802216892607926}

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Input

training_end_year = 2018
auroc_dict = aurocs_training_end_date_2018

training_end_date = f"{training_end_year}-01-01"
evaluation_interval = (
    f"{training_end_year}-01-01",
    f"{training_end_year}-12-31",
)

# Convert dictionary to arrays (sorted)
years = np.array(sorted(auroc_dict.keys()))
x = 2000 + years
y = np.array([auroc_dict[k] for k in years])

# ----- Plot styling -----
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["font.size"] = 11

plt.figure(figsize=(7, 4), dpi=300)

# Always plot points
plt.scatter(
    x, y,
    s=60,
    facecolors="white",
    edgecolors="black",
    linewidth=1.2
)

# ----- Only fit model if >= 2 points -----
if len(x) >= 2:
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    pred = model.get_prediction(X)
    pred_ci = pred.conf_int()
    pred_mean = pred.predicted_mean

    # # Regression line
    # plt.plot(x, pred_mean, color="darkblue", linewidth=1.8)

    # # Confidence interval
    # plt.fill_between(
    #     x,
    #     pred_ci[:, 0],
    #     pred_ci[:, 1],
    #     color="darkblue",
    #     alpha=0.07
    # )

    # slope = model.params[1]
    # slope_se = model.bse[1]

    # print("Slope:", slope)
    # print("Standard error:", slope_se)

else:
    print("Only one observation available; linear model not estimated.")

# Labels and limits
plt.xlabel("Year", fontsize=12)
plt.ylabel("AUROC", fontsize=12)
plt.ylim(0, 1)
plt.xlim(2017.7, 2023.3)

# Grid
plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)


########

zoom_xlim = (2017.7, 2023.3)
zoom_ylim = (0.75, 0.91)


# ----- Draw dashed zoom box on main plot -----
rect = plt.Rectangle(
    (zoom_xlim[0], zoom_ylim[0]),
    zoom_xlim[1] - zoom_xlim[0],
    zoom_ylim[1] - zoom_ylim[0],
    linewidth=1.0,
    edgecolor="darkgray",
    facecolor="none",
    linestyle="--"
)
plt.gca().add_patch(rect)

# ----- Inset axes (zoomed view) -----
ax_main = plt.gca()

ax_inset = inset_axes(
    ax_main,
    width="35%",
    height="35%",
    loc="lower center",
    bbox_to_anchor=(0, 0.08, 1, 1),  # ↑ move inset up
    bbox_transform=ax_main.transAxes,
    borderpad=1.2
)

# Scatter points
ax_inset.scatter(
    x, y,
    s=35,
    facecolors="white",
    edgecolors="black",
    linewidth=1.0
)

# Regression + CI
if len(x) >= 2:
    ax_inset.plot(x, pred_mean, color="darkblue", linewidth=1.2)
    ax_inset.fill_between(
        x,
        pred_ci[:, 0],
        pred_ci[:, 1],
        color="darkblue",
        alpha=0.07
    )

# Zoom limits
ax_inset.set_xlim(*zoom_xlim)
ax_inset.set_ylim(*zoom_ylim)

# Styling
ax_inset.tick_params(axis="both", labelsize=8)
for spine in ax_inset.spines.values():
    spine.set_linewidth(0.8)

ax_inset.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

# ----- Connector lines between main plot and inset -----
mark_inset(
    ax_main,
    ax_inset,
    loc1=2,   # upper left
    loc2=4,   # lower right
    fc="none",
    ec="darkgray",
    linewidth=0.8,
    alpha=0.5,
)

########


plt.tight_layout()

outpath = Path(".")
plt.savefig(
    outpath / f"t2d-extended_linear-regression_inset_{training_end_date}_{evaluation_interval[0]}_{evaluation_interval[1]}.png"
)
plt.show()






##########################################################################################
##########################################################################################
##########################################################################################


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---- Input dictionaries ----
aurocs_by_training_end = {
    2018: {18: 0.8376074504993402, 19: 0.8331785757189495, 20: 0.8716330715094136,
           21: 0.7956406780790461, 22: 0.8399708928832673, 23: 0.7985506465746504},
    2019: {19: 0.8364743910231012, 20: 0.8613122432405681,
           21: 0.7872254209668531, 22: 0.8276700817938417, 23: 0.7896403901045189},
    2020: {20: 0.8670152208419689, 21: 0.7928056922527461,
           22: 0.8338902756140362, 23: 0.7923104728996585},
    2021: {21: 0.8009282114806378, 22: 0.8406449489498253,
           23: 0.7943600358218161},
    2022: {22: 0.8490573397279086, 23: 0.7975422369165425},
    2023: {23: 0.802216892607926},
}

# ---- Flatten data while keeping source label ----
x_all = []
y_all = []
is_2018 = []

for training_end_year, d in aurocs_by_training_end.items():
    for year_suffix, auroc in d.items():
        x_all.append(2000 + year_suffix)
        y_all.append(auroc)
        is_2018.append(training_end_year == 2018)

x = np.array(x_all)
y = np.array(y_all)
is_2018 = np.array(is_2018)

# ---- Sort by calendar year ----
order = np.argsort(x)
x = x[order]
y = y[order]
is_2018 = is_2018[order]

# ---- Plot styling ----
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["font.size"] = 11

plt.figure(figsize=(7, 4), dpi=300)

# ---- Plot 2018-training points (distinct) ----
plt.scatter(
    x[is_2018], y[is_2018],
    s=55,
    facecolors="gray",
    edgecolors="black",
    linewidth=1.1,
)

# ---- Plot all other points ----
plt.scatter(
    x[~is_2018], y[~is_2018],
    s=50,
    facecolors="white",
    edgecolors="black",
    linewidth=1.1,
)

# ---- Overall regression ----
if len(x) >= 2:
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    pred = model.get_prediction(X)
    pred_ci = pred.conf_int()
    pred_mean = pred.predicted_mean

    plt.plot(x, pred_mean, color="darkblue", linewidth=1.8)
    plt.fill_between(
        x, pred_ci[:, 0], pred_ci[:, 1],
        color="darkblue", alpha=0.07
    )

# ---- Axes ----
plt.xlabel("Year", fontsize=12)
plt.ylabel("AUROC", fontsize=12)
plt.ylim(0.78, 0.89)
plt.xlim(2017.7, 2023.3)

plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
plt.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig("t2d-extended_linear-regression_all-years.png")
plt.show()



##########################################################################################
##########################################################################################
##########################################################################################



aurocs_calibrated_only = {18: 0.8376074504993402, 19: 0.8364743910231012, 20: 0.8670152208419689, 21: 0.8009282114806378, 22: 0.8490573397279086, 23: 0.802216892607926}

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

auroc_dict = aurocs_calibrated_only

# Convert dictionary to arrays (sorted)
years = np.array(sorted(auroc_dict.keys()))
x = 2000 + years
y = np.array([auroc_dict[k] for k in years])

# ----- Plot styling -----
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["font.size"] = 11

plt.figure(figsize=(7, 4), dpi=300)

# Always plot points
plt.scatter(
    x, y,
    s=60,
    facecolors="white",
    edgecolors="black",
    linewidth=1.2
)

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

pred = model.get_prediction(X)
pred_ci = pred.conf_int()
pred_mean = pred.predicted_mean

# Regression line
plt.plot(x, pred_mean, color="black", linewidth=1.8)

# Confidence interval
plt.fill_between(
    x,
    pred_ci[:, 0],
    pred_ci[:, 1],
    color="gray",
    alpha=0.18
)

slope = model.params[1]
slope_se = model.bse[1]

print("Slope:", slope)
print("Standard error:", slope_se)

# Labels and limits
plt.xlabel("Year", fontsize=12)
plt.ylabel("AUROC", fontsize=12)
plt.ylim(0.78, 0.89)
plt.xlim(2017.7, 2023.3)

# Grid
plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

plt.tight_layout()

outpath = Path(".")
plt.savefig(
    outpath / f"t2d-extended_linear-regression_calibrated.png"
)
plt.show()





##########################################################################################
##########################################################################################
##########################################################################################



aurocs_training_end_date_2018 = {18: 0.8376074504993402, 19: 0.8331785757189495, 20: 0.8716330715094136, 21: 0.7956406780790461, 22: 0.8399708928832673, 23: 0.7985506465746504}
aurocs_annual_only = {18: 0.8376074504993402, 19: 0.8364743910231012, 20: 0.8670152208419689, 21: 0.8009282114806378, 22: 0.8490573397279086, 23: 0.802216892607926}
aurocs_oneyear = {18: 0.821, 19: 0.825, 20: 0.876, 21: 0.823, 22: 0.840, 23: 0.833}

# Convert to arrays (sorted)
years = np.array(sorted(aurocs_training_end_date_2018.keys()))
x = 2000 + years

y_2018 = np.array([aurocs_training_end_date_2018[k] for k in years])
y_annual = np.array([aurocs_annual_only[k] for k in years])
y_oneyear = np.array([aurocs_oneyear[k] for k in years])

# ---- Styling ----
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["font.size"] = 11

plt.figure(figsize=(7, 4), dpi=300)

# ---- Plot series ----
plt.plot(
    x, y_2018,
    marker="o",
    linestyle="-",
    color="black",
    linewidth=0.4,
    markersize=6,
    markerfacecolor="white",
    label="Original model"
)

plt.plot(
    x, y_annual,
    marker="o",
    linestyle="-",
    color="gray",
    linewidth=0.4,
    markersize=6,
    markerfacecolor="gray",
    label="Annually updated models"
)

plt.plot(
    x, y_oneyear,
    marker="o",
    linestyle="-",
    color="darkblue",
    linewidth=0.4,
    markersize=6,
    markerfacecolor="darkblue",
    label="One-year models"
)

# ---- Axes ----
plt.xlabel("Year", fontsize=12)
plt.ylabel("AUROC", fontsize=12)
plt.ylim(0.78, 0.89)
plt.xlim(2017.7, 2023.3)

plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
plt.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig("t2d-extended_linear-regression_original_vs_annual_vs_oneyear.png")
plt.show()




##########################################################################################
##########################################################################################
##########################################################################################


aurocs_increasing_window = {13: 0.802, 14: 0.802, 15: 0.802, 16: 0.804, 17: 0.807, 18: 0.809, 19: 0.813, 20: 0.816, 21: 0.815, 22: 0.833}


# Convert to arrays (sorted)
years = np.array(sorted(aurocs_increasing_window.keys()))
x = 2000 + years

y_increasing = np.array([aurocs_increasing_window[k] for k in years])

# ---- Styling ----
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["font.size"] = 11

plt.figure(figsize=(7, 4), dpi=300)

# ---- Plot series ----
plt.plot(
    x, y_increasing,
    marker="o",
    linestyle="-",
    color="black",
    linewidth=0.4,
    markersize=6,
    markerfacecolor="white",
)


# ---- Axes ----
plt.xlabel("Training data interval", fontsize=12)
plt.ylabel("AUROC", fontsize=12)
plt.ylim(0.80, 0.835)
plt.xlim(2012.7, 2022.3)

# ---- Custom x ticks: show every year and label as "X-2022" ----
plt.xticks(
    ticks=x,                                  # all years
    labels=[f"{int(year)}-2022" for year in x],# formatted labels
    rotation=90                               # vertical text
)

plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
plt.legend(frameon=False, fontsize=10)

plt.tight_layout()
#plt.savefig("t2d-extended_linear-regression_increasing.png")
plt.show()














from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# -----------------------------
# Input
# -----------------------------
training_end_year = 2018
auroc_dict = {
    '2018-Q1': 0.8400040336969341,
    '2018-Q2': 0.8359752673163382,
    '2018-Q3': 0.8404948619647145,
    '2018-Q4': 0.8347001040175571,
    '2019-Q1': 0.8209814966488928,
    '2019-Q2': 0.8213330179464475,
    '2019-Q3': 0.8481368713034474,
    '2019-Q4': 0.8464409792346841,
    '2020-Q1': 0.8351541791499517,
    '2020-Q2': 0.8776858827208691,
    '2020-Q3': 0.8985925492795771,
    '2020-Q4': 0.8750314173563778,
    '2021-Q1': 0.8004293018520722,
    '2021-Q2': 0.8173167136163941,
    '2021-Q3': 0.7843003536258585,
    '2021-Q4': 0.789668354178592,
    '2022-Q1': 0.8384457827344268,
    '2022-Q2': 0.8390170790161712,
    '2022-Q3': 0.8396669059806182,
    '2022-Q4': 0.84729069906338,
    '2023-Q1': 0.7985506465746504,
    '2023-Q2': None,
    '2023-Q3': None,
    '2023-Q4': None,
}

training_end_date = f"{training_end_year}-01-01"
evaluation_interval = (
    f"{training_end_year}-01-01",
    f"{training_end_year}-12-31",
)

# -----------------------------
# Parse quarterly dict
# -----------------------------
def quarter_to_decimal_year(label: str) -> float:
    m = re.fullmatch(r"(\d{4})-Q([1-4])", label)
    if not m:
        raise ValueError(f"Unexpected quarter label: {label}")
    year = int(m.group(1))
    quarter = int(m.group(2))
    # place each point in the middle of the quarter
    return year + (quarter - 0.5) / 4

valid_items = [(k, v) for k, v in auroc_dict.items() if v is not None]
valid_items = sorted(valid_items, key=lambda kv: quarter_to_decimal_year(kv[0]))

labels = [k for k, _ in valid_items]
x = np.array([quarter_to_decimal_year(k) for k in labels])
y = np.array([v for _, v in valid_items])

# -----------------------------
# Plot styling
# -----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["font.size"] = 11

fig, ax_main = plt.subplots(figsize=(7, 4), dpi=300)

# main scatter
ax_main.scatter(
    x, y,
    s=40,
    facecolors="white",
    edgecolors="black",
    linewidth=1.0,
    zorder=3
)

# connect quarterly observations
ax_main.plot(x, y, color="black", linewidth=0.9, alpha=0.7, zorder=2)

# -----------------------------
# OLS trend line
# -----------------------------
if len(x) >= 2:
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    x_pred = np.linspace(x.min(), x.max(), 300)
    X_pred = sm.add_constant(x_pred)

    pred = model.get_prediction(X_pred)
    pred_mean = pred.predicted_mean
    pred_ci = pred.conf_int()

    # Uncomment if you want trend line in main panel too
    # ax_main.plot(x_pred, pred_mean, color="darkblue", linewidth=1.5, zorder=1)
    # ax_main.fill_between(
    #     x_pred,
    #     pred_ci[:, 0],
    #     pred_ci[:, 1],
    #     color="darkblue",
    #     alpha=0.07,
    #     zorder=0
    # )

    slope = model.params[1]
    slope_se = model.bse[1]
    print("Slope:", slope)
    print("Standard error:", slope_se)
else:
    print("Only one observation available; linear model not estimated.")

# -----------------------------
# Main axes formatting
# -----------------------------
ax_main.set_xlabel("Year", fontsize=12)
ax_main.set_ylabel("AUROC", fontsize=12)
ax_main.set_ylim(0, 1)
ax_main.set_xlim(2017.7, 2023.3)
ax_main.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

# year ticks only
year_ticks = np.arange(2018, 2024)
ax_main.set_xticks(year_ticks)
ax_main.set_xticklabels([str(y) for y in year_ticks])

# optional vertical lines between years
for year in range(2019, 2024):
    ax_main.axvline(year, color="lightgray", linestyle=":", linewidth=0.6, zorder=0)

# -----------------------------
# Zoom area
# -----------------------------
zoom_xlim = (2017.7, 2023.3)
zoom_ylim = (0.75, 0.91)

rect = plt.Rectangle(
    (zoom_xlim[0], zoom_ylim[0]),
    zoom_xlim[1] - zoom_xlim[0],
    zoom_ylim[1] - zoom_ylim[0],
    linewidth=1.0,
    edgecolor="darkgray",
    facecolor="none",
    linestyle="--"
)
ax_main.add_patch(rect)

# -----------------------------
# Inset
# -----------------------------
ax_inset = inset_axes(
    ax_main,
    width="35%",
    height="35%",
    loc="lower center",
    bbox_to_anchor=(0, 0.08, 1, 1),
    bbox_transform=ax_main.transAxes,
    borderpad=1.2
)

ax_inset.scatter(
    x, y,
    s=22,
    facecolors="white",
    edgecolors="black",
    linewidth=0.8,
    zorder=3
)
ax_inset.plot(x, y, color="black", linewidth=0.8, alpha=0.7, zorder=2)

if len(x) >= 2:
    ax_inset.plot(x_pred, pred_mean, color="darkblue", linewidth=1.2, zorder=1)
    ax_inset.fill_between(
        x_pred,
        pred_ci[:, 0],
        pred_ci[:, 1],
        color="darkblue",
        alpha=0.07,
        zorder=0
    )

ax_inset.set_xlim(*zoom_xlim)
ax_inset.set_ylim(*zoom_ylim)
ax_inset.set_xticks(year_ticks)
ax_inset.set_xticklabels([str(y) for y in year_ticks], fontsize=8)
ax_inset.tick_params(axis="y", labelsize=8)

for spine in ax_inset.spines.values():
    spine.set_linewidth(0.8)

ax_inset.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

mark_inset(
    ax_main,
    ax_inset,
    loc1=2,
    loc2=4,
    fc="none",
    ec="darkgray",
    linewidth=0.8,
    alpha=0.5,
)

# -----------------------------
# Save
# -----------------------------
plt.tight_layout()

outpath = Path(".")
plt.savefig(
    outpath / f"t2d-extended_quarterly-regression_inset_{training_end_date}_{evaluation_interval[0]}_{evaluation_interval[1]}.png",
    bbox_inches="tight"
)
plt.show()
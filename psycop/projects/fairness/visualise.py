import plotnine as pn
import pandas as pd

if __name__ == "__main__":
        
    df = pd.read_csv("metrics_sex.csv")
    df = df.drop(columns="Unnamed: 0")
    
    palette = ["#6D7A5F", "#D8973C", "#983D4E", "#52657A", "#B8683A", "#985D98"]

    df_auc = df[df["variable"] == "False positive rate"]
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as mtick

    # Example setup
    models = df_auc["model"].unique()
    sexes = df_auc["sex"].unique()

    x = np.arange(len(models))
    width = 0.35

    # HERE
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Secondary axis
    ax2 = ax1.twinx()

    # Plot separately by sex
    for i, sex in enumerate(sexes):

        sub = df_auc[df_auc["sex"] == sex]

        xpos = x + (i - 0.5) * width

        # --- Bars: proportion ---
        ax2.bar(
            xpos,
            sub["proportion"],
            width=width,
            alpha=0.25,
            color=palette[i],
            label=f"{sex} proportion"
        )

        # --- Points + error bars: False positive rate ---
        ax1.errorbar(
            xpos,
            sub["value"],
            yerr=[
                sub["value"] - sub["lower"],
                sub["upper"] - sub["value"]
            ],
            fmt="o",
            color="black",
            capsize=3,
            label=f"{sex} False positive rate"
        )

    # X-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.set_xlabel("Model")

    # Y-axes
    ax1.grid(True, alpha=0.3)

    # Independent limits
    ax1.set_ylim(0, 0.08)
    ax2.set_ylim(0.0, 1)
    ax2.yaxis.label.set_alpha(0.6)
    ax1.tick_params(axis="both", labelsize=11)
    ax2.tick_params(axis="both", colors=(0, 0, 0, 0.5), labelsize=11)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    # Title
    ax1.set_title("False positive rate", size=20)
    ax1.set_xlabel("Model", fontsize=14)
    ax1.set_ylabel("FPR", fontsize=14)
    ax2.set_ylabel("Percentage", fontsize=14)

    # Spines
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("plt.png")











    (
        pn.ggplot(df[df["variable"] == "AUROC"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", color="sex"))
        + pn.geom_bar(pn.aes(y="proportion", fill="sex"), stat="identity", position="dodge", alpha=0.6, width=0.7)
        + pn.geom_errorbar(width = 0.4, position = pn.position_dodge(width = 0.7))
        + pn.geom_point(size=1.5, position = pn.position_dodge(width = 0.7))
        + pn.labs(x="Model", y="AUROC / Proportion", title="AUROC", fill="Sex")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            #figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
        + pn.scale_color_manual(values=["black", "black"], guide=False)
        + pn.coord_cartesian(ylim=(0.25, 1))
    ).save("test.png")

    (
        pn.ggplot(df[df["variable"] == "True positive rate"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", color="sex"))
        + pn.geom_bar(pn.aes(y="proportion", fill="sex"), stat="identity", position="dodge", alpha=0.6, width=0.7)
        + pn.geom_errorbar(width = 0.4, position = pn.position_dodge(width = 0.7))
        + pn.geom_point(size=1.5, position = pn.position_dodge(width = 0.7))
        + pn.labs(x="Model", y="TPR / Proportion", title="True positive rate", fill="Sex")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            #figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
        + pn.scale_color_manual(values=["black", "black"], guide=False)
        + pn.coord_cartesian(ylim=(0.0, 0.75))
    ).save("test.png")

    (
        pn.ggplot(df[df["variable"] == "True negative rate"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", color="sex"))
        + pn.geom_bar(pn.aes(y="proportion", fill="sex"), stat="identity", position="dodge", alpha=0.6, width=0.7)
        + pn.geom_errorbar(width = 0.4, position = pn.position_dodge(width = 0.7))
        + pn.geom_point(size=1.5, position = pn.position_dodge(width = 0.7))
        + pn.labs(x="Model", y="TNR / Proportion", title="True negative rate", fill="Sex")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            #figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
        + pn.scale_color_manual(values=["black", "black"], guide=False)
        + pn.coord_cartesian(ylim=(0.25, 1))
    ).save("test.png")

    (
        pn.ggplot(df[df["variable"] == "False positive rate"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", color="sex"))
        + pn.geom_bar(pn.aes(y="proportion", fill="sex"), stat="identity", position="dodge", alpha=0.6, width=0.7)
        + pn.geom_errorbar(width = 0.4, position = pn.position_dodge(width = 0.7))
        + pn.geom_point(size=1.5, position = pn.position_dodge(width = 0.7))
        + pn.labs(x="Model", y="FPR / Proportion", title="False positive rate", fill="Sex")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            #figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
        + pn.scale_color_manual(values=["black", "black"], guide=False)
        + pn.coord_cartesian(ylim=(0, 0.75))
    ).save("test.png")

    (
        pn.ggplot(df[df["variable"] == "False negative rate"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", color="sex"))
        + pn.geom_bar(pn.aes(y="proportion", fill="sex"), stat="identity", position="dodge", alpha=0.6, width=0.7)
        + pn.geom_errorbar(width = 0.4, position = pn.position_dodge(width = 0.7))
        + pn.geom_point(size=1.5, position = pn.position_dodge(width = 0.7))
        + pn.labs(x="Model", y="FNR / Proportion", title="False negative rate", fill="Sex")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            #figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
        + pn.scale_color_manual(values=["black", "black"], guide=False)
        + pn.coord_cartesian(ylim=(0.25, 1))
    ).save("test.png")

    (
        pn.ggplot(df[df["variable"] == "Selection rate"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", color="sex"))
        + pn.geom_bar(pn.aes(y="proportion", fill="sex"), stat="identity", position="dodge", alpha=0.6, width=0.7)
        + pn.geom_errorbar(width = 0.4, position = pn.position_dodge(width = 0.7))
        + pn.geom_point(size=1.5, position = pn.position_dodge(width = 0.7))
        + pn.labs(x="Model", y="Selection rate / Proportion", title="Selection rate", fill="Sex")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            #figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
        + pn.scale_color_manual(values=["black", "black"], guide=False)
        + pn.coord_cartesian(ylim=(0, 0.75))
    ).save("test.png")

    (
        pn.ggplot(df[df["variable"] == "Positive predictive value"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", color="sex"))
        + pn.geom_bar(pn.aes(y="proportion", fill="sex"), stat="identity", position="dodge", alpha=0.6, width=0.7)
        + pn.geom_errorbar(width = 0.4, position = pn.position_dodge(width = 0.7))
        + pn.geom_point(size=1.5, position = pn.position_dodge(width = 0.7))
        + pn.labs(x="Model", y="PPV / Proportion", title="Positive predictive value", fill="Sex")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            #figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
        + pn.scale_color_manual(values=["black", "black"], guide=False)
        + pn.coord_cartesian(ylim=(0, 0.75))
    ).save("test.png")

    import matplotlib.pyplot as plt
    import numpy as np

    

    fig, ax = plt.subplots()
    categories = np.unique(df["sex"])
    colordict = dict(zip(categories, palette))  

    df["Color"] = df["sex"].apply(lambda x: colordict[x])

    ax.errorbar(df[df["variable"] == "AUROC"]["model"], df[df["variable"] == "AUROC"]["value"], yerr=[df[df["variable"] == "AUROC"]["lower"], df[df["variable"] == "AUROC"]["upper"]], c="black", fmt="o", capsize=5)

    fig.savefig("test_plt.png")







    (
        pn.ggplot(df[df["variable"] == "AUROC"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", fill="age"))
        + pn.stat_summary(geom="bar", position="dodge")
        + pn.stat_summary_bin(geom="errorbar", position="dodge")
        + pn.labs(x="Model", y="AUROC", title="AUROC", fill="Age")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            #axis_title=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            # figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
        + pn.coord_cartesian(ylim=(0.4, 1))
    ).save("test.png")

    (
        pn.ggplot(df[df["variable"] == "True positive rate"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", fill="sex"))
        + pn.stat_summary(geom="bar", position="dodge")
        + pn.stat_summary_bin(geom="errorbar", position="dodge")
        + pn.labs(x="Model", y="True positive rate", title="True positive rate", fill="Sex")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            #axis_title=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            # figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
        + pn.coord_cartesian(ylim=(0, 0.75))
    ).save("test.png")

    (
        pn.ggplot(df[df["variable"] == "True positive rate"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", fill="age"))
        + pn.stat_summary(geom="bar", position="dodge")
        + pn.stat_summary_bin(geom="errorbar", position="dodge")
        + pn.labs(x="Model", y="True positive rate", title="True positive rate", fill="Age")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            #axis_title=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            # figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
    ).save("test.png")

    (
        pn.ggplot(df[df["variable"] == "True negative rate"], pn.aes(x="model", y="value", ymin="lower", ymax="upper", fill="sex"))
        + pn.stat_summary(geom="bar", position="dodge")
        + pn.stat_summary_bin(geom="errorbar", position="dodge")
        + pn.labs(x="Model", y="True negative rate", title="True negative rate", fill="Sex")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            #axis_title=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            # figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
        + pn.coord_cartesian(ylim=(0.5, 1))
    ).save("test.png")

    (
        pn.ggplot(df[(df["variable"] == "True negative rate") & (df["model"] != "CVD")], pn.aes(x="model", y="value", ymin="lower", ymax="upper", fill="sex"))
        + pn.stat_summary(geom="bar", position="dodge")
        + pn.stat_summary_bin(geom="errorbar", position="dodge")
        + pn.labs(x="Model", y="True negative rate", title="True negative rate", fill="Sex")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            #axis_title=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            # figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
        + pn.coord_cartesian(ylim=(0.88, 1))
    ).save("test.png")

    (
        pn.ggplot(df[(df["variable"] == "True negative rate") & (df["model"] != "CVD")], pn.aes(x="model", y="value", ymin="lower", ymax="upper", fill="age"))
        + pn.stat_summary(geom="bar", position="dodge")
        + pn.stat_summary_bin(geom="errorbar", position="dodge")
        + pn.labs(x="Model", y="True negative rate", title="True negative rate", fill="Age")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45),
            panel_grid_minor=pn.element_blank(),
            #axis_title=pn.element_blank(),
            plot_title=pn.element_text(size=20, ha="center"),
            dpi=300,
            # figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=palette)
        + pn.coord_cartesian(ylim=(0.78, 1))
    ).save("test.png")


    pass
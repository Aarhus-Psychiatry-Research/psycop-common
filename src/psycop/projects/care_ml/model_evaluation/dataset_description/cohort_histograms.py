import pandas as pd
import plotnine as pn
from care_ml.model_evaluation.config import (
    COLOURS,
    GENERAL_ARTIFACT_PATH,
    PN_THEME,
)
from care_ml.model_evaluation.dataset_description.utils import (
    get_sex_and_age_group_at_first_contact,
    load_feature_set,
)


def main():
    # load train and test splits
    df = load_feature_set()

    # sex and age histogram
    sex_age = get_sex_and_age_group_at_first_contact(df).drop(
        columns=["dw_ek_borger", "pred_age_in_years"],
    )
    sex_age = pd.concat(
        [sex_age.value_counts("pred_sex_female"), sex_age.value_counts("age_group")],
    ).reset_index()
    sex_age.loc[sex_age["index"] == False, "index"] = "Male"  # noqa: E712
    sex_age.loc[sex_age["index"] == True, "index"] = "Female"  # noqa: E712
    sex_age.columns = ["Category", "Value"]

    sex_age["Category"] = pd.Categorical(
        sex_age["Category"],
        categories=[
            "Female",
            "Male",
            "18-20",
            "21-25",
            "26-30",
            "31-35",
            "36-40",
            "41-45",
            "46-50",
            "51-55",
            "56-60",
            "61-65",
            "66-70",
            "71-75",
            "76+",
        ],
    )
    sex_age = sex_age.sort_values("Category")

    sex = sex_age[(sex_age["Category"] == "Female") | (sex_age["Category"] == "Male")]
    age = sex_age[(sex_age["Category"] != "Female") & (sex_age["Category"] != "Male")]

    dodge_text = pn.position_dodge(width=0.9)

    # sex
    (
        pn.ggplot(sex, pn.aes(x="Category", y="Value"))
        + pn.geom_col(stat="identity", width=0.9, fill=COLOURS["blue"])
        + pn.geom_text(
            pn.aes(label="Value"),
            position=pn.position_dodge(width=0.9),
            size=16,
            va="bottom",
        )
        + pn.lims(y=(0, 8000))
        + pn.labs(y="Patients (n)", x="Sex", title="A) Sex Distribution")
        + pn.scale_fill_manual(
            values=[
                COLOURS["blue"],
                COLOURS["green"],
            ],
        )
        + PN_THEME
        + pn.theme(figure_size=(3, 5), text=pn.element_text(size=20))
    ).save(GENERAL_ARTIFACT_PATH.parent.parent / "cohort_hist_sex.png", dpi=600)

    (
        pn.ggplot(age, pn.aes(x="Category", y="Value"))
        + pn.geom_col(stat="identity", width=0.9, fill=COLOURS["blue"])
        + pn.geom_text(pn.aes(label="Value"), position=dodge_text, size=16, va="bottom")
        + pn.lims(y=(0, 2200))
        + pn.labs(
            y="Patients (n)",
            x="Age Group",
            title="B) Age Distribution (At First Contact)",
        )
        + PN_THEME
        + pn.theme(figure_size=(13, 5), text=pn.element_text(size=20))
    ).save(GENERAL_ARTIFACT_PATH.parent.parent / "cohort_hist_age_group.png", dpi=600)

    # admission length histogram
    adm_length = df.groupby(["adm_id"])["pred_adm_day_count"].max()
    adm_length = pd.DataFrame(adm_length).reset_index(drop=False)
    adm_length = adm_length.value_counts("pred_adm_day_count")
    adm_length = pd.DataFrame(adm_length).reset_index(drop=False)
    adm_length.columns = ["Day of Admission", "Count"]
    adm_length = adm_length.sort_values("Day of Admission")

    bins = [0, 10, 20, 30, 40, 50, 60, 70]
    labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
    adm_length["outcome_bins"] = pd.cut(
        adm_length["Day of Admission"],
        bins=bins,
        labels=labels,
        right=False,
    )

    adm_length = adm_length.groupby("outcome_bins").sum().reset_index()
    (
        pn.ggplot(
            adm_length,
            pn.aes(x="outcome_bins", y="Count"),
        )  # , color=COLOURS["blue"]))
        + pn.geom_col(stat="identity", width=0.9, fill=COLOURS["blue"])
        + pn.lims(y=(0, 26000))
        + pn.labs(
            y="Admissions (n)",
            x="Number of Days in Admission",
            title="C) Distribution of Admission Lengths",
        )
        + pn.geom_text(pn.aes(label="Count"), va="bottom")
        + PN_THEME
        + pn.theme(figure_size=(15, 5), text=pn.element_text(size=20))
    ).save(GENERAL_ARTIFACT_PATH.parent.parent / "cohort_adm_length_hist.png", dpi=600)

    # day of outcome
    day_of_outcome = df[df["outcome_timestamp"].notnull()][
        ["adm_id", "outcome_timestamp", "pred_adm_day_count"]
    ]
    day_of_outcome = df.groupby(["adm_id", "outcome_timestamp"])[
        "pred_adm_day_count"
    ].max()
    day_of_outcome = pd.DataFrame(day_of_outcome).reset_index()

    bins = [0, 10, 20, 30, 40, 50, 60, 70]
    labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
    day_of_outcome["outcome_bins"] = pd.cut(
        day_of_outcome["pred_adm_day_count"],
        bins=bins,
        labels=labels,
        right=False,
    )
    day_of_outcome = day_of_outcome[["adm_id", "outcome_bins"]].value_counts(
        "outcome_bins",
    )
    day_of_outcome = pd.DataFrame(day_of_outcome).reset_index(drop=False)
    day_of_outcome.columns = ["Outcome bins", "Count"]
    day_of_outcome = day_of_outcome.sort_values("Outcome bins")

    (
        pn.ggplot(
            day_of_outcome,
            pn.aes(x="Outcome bins", y="Count"),
        )  # , color=COLOURS["blue"]))
        + pn.geom_col(stat="identity", width=0.9, fill=COLOURS["blue"])
        + pn.labs(
            y="Outcomes (n)",
            x="Day in Admission",
            title="D) Distribution of Admission Days with Outcome",
        )
        + pn.geom_text(pn.aes(label="Count"), size=14, va="bottom")
        + pn.ylim(0, 1100)
        + PN_THEME
        + pn.theme(figure_size=(15, 5), text=pn.element_text(size=20))
    ).save(GENERAL_ARTIFACT_PATH.parent.parent / "cohort_outcome_bins.png", dpi=600)


if __name__ == "__main__":
    main()

"""util functions to create table one"""
import pandas as pd
from psycop.common.model_training.application_modules.process_manager_setup import setup
from psycop.common.model_training.data_loader.data_loader import DataLoader


def load_feature_set() -> pd.DataFrame:
    """Loads the feature set with text features

    Returns:
        pd.DataFrame: Df with column denoting train or test
    """
    cfg, _ = setup(
        config_file_name="default_config.yaml",
        application_config_dir_relative_path="../../../../../../psycop_coercion/model_training/application/config/",
    )

    train_df = DataLoader(data_cfg=cfg.data).load_dataset_from_dir(split_names="train")
    test_df = DataLoader(data_cfg=cfg.data).load_dataset_from_dir(split_names="val")

    train_df["dataset"] = "train"
    test_df["dataset"] = "test"

    return pd.concat([train_df, test_df])


def table_one_coercion(df: pd.DataFrame) -> pd.DataFrame:
    """Create table one - target days and coercion instances

    Args:
        df (pd.DataFrame): Train and test set concatenated

    Returns:
        pd.DataFrame: Table one - coercion
    """

    target_days = df[
        [
            "adm_id",
            "dw_ek_borger",
            "outcome_timestamp",
            "pred_adm_day_count",
            "outcome_coercion_bool_within_2_days",
            "dataset",
        ]
    ][df["outcome_coercion_bool_within_2_days"] == 1].reset_index()

    coercion_instances = (
        target_days.groupby(["adm_id", "dw_ek_borger", "outcome_timestamp", "dataset"])[
            "pred_adm_day_count"
        ]
        .max()
        .reset_index()
    )

    day_of_outcome = df[df["outcome_timestamp"].notnull()][
        ["adm_id", "outcome_timestamp", "pred_adm_day_count"]
    ]
    day_of_outcome = df.groupby(["adm_id", "outcome_timestamp"])[
        "pred_adm_day_count"
    ].max()
    day_of_outcome = pd.DataFrame(day_of_outcome).reset_index()

    # Patients w coercion
    pt_w_coercion = len(coercion_instances["dw_ek_borger"].unique())
    pt_w_coercion_p = round(pt_w_coercion / len(df["dw_ek_borger"].unique()) * 100, 2)

    pt_w_coercion_train = len(
        coercion_instances[coercion_instances["dataset"] == "train"][
            "dw_ek_borger"
        ].unique(),
    )
    pt_w_coercion_p_train = round(
        pt_w_coercion_train
        / len(df[df["dataset"] == "train"]["dw_ek_borger"].unique())
        * 100,
        2,
    )

    pt_w_coercion_test = len(
        coercion_instances[coercion_instances["dataset"] == "test"][
            "dw_ek_borger"
        ].unique(),
    )
    pt_w_coercion_p_test = round(
        pt_w_coercion_test
        / len(df[df["dataset"] == "test"]["dw_ek_borger"].unique())
        * 100,
        2,
    )

    pts_with_coercion = pd.DataFrame(
        {
            "Grouping": "Patients with outcome, n (%)",
            "Overall": f"{pt_w_coercion} ({pt_w_coercion_p})",
            "Train": f"{pt_w_coercion_train} ({pt_w_coercion_p_train})",
            "Test": f"{pt_w_coercion_test} ({pt_w_coercion_p_test})",
        },
        index=[0],
    )

    # Adms with coercion
    adms_w_coercion = len(coercion_instances["adm_id"].unique())
    adms_w_coercion_p = round(adms_w_coercion / len(df["adm_id"].unique()) * 100, 2)

    adms_w_coercion_train = len(
        coercion_instances[coercion_instances["dataset"] == "train"]["adm_id"].unique(),
    )
    adms_w_coercion_p_train = round(
        adms_w_coercion_train
        / len(df[df["dataset"] == "train"]["adm_id"].unique())
        * 100,
        2,
    )

    adms_w_coercion_test = len(
        coercion_instances[coercion_instances["dataset"] == "test"]["adm_id"].unique(),
    )
    adms_w_coercion_p_test = round(
        adms_w_coercion_test
        / len(df[df["dataset"] == "test"]["adm_id"].unique())
        * 100,
        2,
    )

    adms_with_coercion = pd.DataFrame(
        {
            "Grouping": "Admissions with outcome, n (%)",
            "Overall": f"{adms_w_coercion} ({adms_w_coercion_p})",
            "Train": f"{adms_w_coercion_train} ({adms_w_coercion_p_train})",
            "Test": f"{adms_w_coercion_test} ({adms_w_coercion_p_test})",
        },
        index=[0],
    )

    # Median day coercion happens in adms
    coercion_day_in_adm_m = coercion_instances["pred_adm_day_count"].median()
    coercion_day_in_adm_Q1 = coercion_instances["pred_adm_day_count"].quantile(q=0.25)
    coercion_day_in_adm_Q3 = coercion_instances["pred_adm_day_count"].quantile(q=0.75)

    coercion_day_in_adm_m_train = coercion_instances[
        coercion_instances["dataset"] == "train"
    ]["pred_adm_day_count"].median()
    coercion_day_in_adm_Q1_train = coercion_instances[
        coercion_instances["dataset"] == "train"
    ]["pred_adm_day_count"].quantile(q=0.25)
    coercion_day_in_adm_Q3_train = coercion_instances[
        coercion_instances["dataset"] == "train"
    ]["pred_adm_day_count"].quantile(q=0.75)

    coercion_day_in_adm_m_test = coercion_instances[
        coercion_instances["dataset"] == "test"
    ]["pred_adm_day_count"].median()
    coercion_day_in_adm_Q1_test = coercion_instances[
        coercion_instances["dataset"] == "test"
    ]["pred_adm_day_count"].quantile(q=0.25)
    coercion_day_in_adm_Q3_test = coercion_instances[
        coercion_instances["dataset"] == "test"
    ]["pred_adm_day_count"].quantile(q=0.75)

    coercion_day_in_adm = pd.DataFrame(
        {
            "Grouping": "Admission day with outcome, median (Q1, Q3)",
            "Overall": f"{coercion_day_in_adm_m} ({coercion_day_in_adm_Q1}, {coercion_day_in_adm_Q3})",
            "Train": f"{coercion_day_in_adm_m_train} ({coercion_day_in_adm_Q1_train}, {coercion_day_in_adm_Q3_train})",
            "Test": f"{coercion_day_in_adm_m_test} ({coercion_day_in_adm_Q1_test}, {coercion_day_in_adm_Q3_test})",
        },
        index=[0],
    )

    # Pred days with target
    days_w_target = len(target_days)
    days_w_target_p = round(days_w_target / len(df) * 100, 2)

    days_w_target_train = len(target_days[target_days["dataset"] == "train"])
    days_w_target_p_train = round(
        days_w_target_train / len(df[df["dataset"] == "train"]) * 100,
        2,
    )

    days_w_target_test = len(target_days[target_days["dataset"] == "test"])
    days_w_target_p_test = round(
        days_w_target_test / len(df[df["dataset"] == "test"]) * 100,
        2,
    )

    days_w_target = pd.DataFrame(
        {
            "Grouping": "Target days, n (%)",
            "Overall": f"{days_w_target} ({days_w_target_p})",
            "Train": f"{days_w_target_train} ({days_w_target_p_train})",
            "Test": f"{days_w_target_test} ({days_w_target_p_test})",
        },
        index=[0],
    )

    # Pred days with coercion
    days_w_coercion = len(coercion_instances)
    days_w_coercion_p = round(days_w_coercion / len(df) * 100, 2)

    days_w_coercion_train = len(
        coercion_instances[coercion_instances["dataset"] == "train"],
    )
    days_w_coercion_p_train = round(
        days_w_coercion_train / len(df[df["dataset"] == "train"]) * 100,
        2,
    )

    days_w_coercion_test = len(
        coercion_instances[coercion_instances["dataset"] == "test"],
    )
    days_w_coercion_p_test = round(
        days_w_coercion_test / len(df[df["dataset"] == "test"]) * 100,
        2,
    )

    days_w_coercion = pd.DataFrame(
        {
            "Grouping": "Days with coercion, n (%)",
            "Overall": f"{days_w_coercion} ({days_w_coercion_p})",
            "Train": f"{days_w_coercion_train} ({days_w_coercion_p_train})",
            "Test": f"{days_w_coercion_test} ({days_w_coercion_p_test})",
        },
        index=[0],
    )

    # Collect it all
    return pd.concat(
        [
            pts_with_coercion,
            adms_with_coercion,
            coercion_day_in_adm,
            days_w_target,
            days_w_coercion,
        ],
    )


def table_one_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Create table one - demographics

    Args:
        df (pd.DataFrame): Train and test set

    Returns:
        pd.DataFrame: Table one - demographics
    """

    # demographics
    dem = df[["dw_ek_borger", "pred_sex_female", "pred_age_in_years", "dataset"]]
    dem = (
        dem.groupby(["dw_ek_borger", "pred_sex_female", "dataset"])["pred_age_in_years"]
        .min()
        .reset_index()
    )

    age_groups = [18, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 120]

    labels = [
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
    ]

    dem["age_group"] = pd.cut(
        dem["pred_age_in_years"],
        bins=age_groups,
        labels=labels,
        right=False,
    )

    # n prediction days
    n_prediction_days = pd.DataFrame(
        {
            "Grouping": "Prediction days, n",
            "Overall": df.shape[0],
            "Train": df[df["dataset"] == "train"].shape[0],
            "Test": df[df["dataset"] == "test"].shape[0],
        },
        index=[0],
    )

    # n admissions
    n_adms = pd.DataFrame(
        {
            "Grouping": "Admissions, n",
            "Overall": len(df["adm_id"].unique()),
            "Train": len(df[df["dataset"] == "train"]["adm_id"].unique()),
            "Test": len(df[df["dataset"] == "test"]["adm_id"].unique()),
        },
        index=[0],
    )

    # patients
    n_pts = pd.DataFrame(
        {
            "Grouping": "Patients, n",
            "Overall": len(dem["dw_ek_borger"]),
            "Train": len(dem[dem["dataset"] == "train"]["dw_ek_borger"]),
            "Test": len(dem[dem["dataset"] == "test"]["dw_ek_borger"]),
        },
        index=[0],
    )

    # sex
    sex_n = len(dem[dem["pred_sex_female"] == True])  # noqa: E712
    sex_p = round(
        len(dem[dem["pred_sex_female"] == True])  # noqa: E712
        / len(dem["dw_ek_borger"].unique())
        * 100,
        2,
    )
    sex_train_n = len(
        dem[
            (dem["pred_sex_female"] == True) & (dem["dataset"] == "train")  # noqa: E712
        ],
    )
    sex_train_p = round(
        len(
            dem[
                (dem["pred_sex_female"] == True)  # noqa: E712
                & (dem["dataset"] == "train")
            ],
        )
        / len(dem[dem["dataset"] == "train"]["dw_ek_borger"])
        * 100,
        2,
    )
    sex_test_n = len(
        dem[
            (dem["pred_sex_female"] == True) & (dem["dataset"] == "test")  # noqa: E712
        ],
    )
    sex_test_p = round(
        len(
            dem[
                (dem["pred_sex_female"] == True)  # noqa: E712
                & (dem["dataset"] == "test")
            ],
        )
        / len(dem[dem["dataset"] == "test"]["dw_ek_borger"])
        * 100,
        2,
    )

    n_sex_female = pd.DataFrame(
        {
            "Grouping": "Sex female, n (%)",
            "Overall": f"{sex_n} ({sex_p})",
            "Train": f"{sex_train_n} ({sex_train_p})",
            "Test": f"{sex_test_n} ({sex_test_p})",
        },
        index=[0],
    )

    # age median
    train_m = dem[dem["dataset"] == "train"]["pred_age_in_years"].median()
    train_q1 = dem[dem["dataset"] == "train"]["pred_age_in_years"].quantile(q=0.25)
    train_q3 = dem[dem["dataset"] == "train"]["pred_age_in_years"].quantile(q=0.75)

    test_m = dem[dem["dataset"] == "test"]["pred_age_in_years"].median()
    test_q1 = dem[dem["dataset"] == "test"]["pred_age_in_years"].quantile(q=0.25)
    test_q3 = dem[dem["dataset"] == "test"]["pred_age_in_years"].quantile(q=0.75)

    age_median = pd.DataFrame(
        {
            "Grouping": "Age, median (Q1, Q3)",
            "Overall": f"{dem.pred_age_in_years.median()} ({dem.pred_age_in_years.quantile(q=0.25)}, {dem.pred_age_in_years.quantile(q=0.75)})",
            "Train": f"{train_m} ({train_q1}, {train_q3})",
            "Test": f"{test_m} ({test_q1}, {test_q3})",
        },
        index=[0],
    )

    # age groups
    ag_overall = pd.DataFrame(
        {
            "Overall_n": dem["age_group"].value_counts(),
            "Overall_%": round(
                (dem["age_group"].value_counts() / len(dem["dw_ek_borger"])) * 100,
                2,
            ),
        },
    )
    ag_overall["Overall"] = (
        ag_overall["Overall_n"].astype(str)
        + " ("
        + ag_overall["Overall_%"].astype(str)
        + ")"
    )

    ag_train = pd.DataFrame(
        {
            "Train_n": dem[dem["dataset"] == "train"]["age_group"].value_counts(),
            "Train_%": round(
                (
                    dem[dem["dataset"] == "train"]["age_group"].value_counts()
                    / len(dem[dem["dataset"] == "train"]["dw_ek_borger"])
                )
                * 100,
                2,
            ),
        },
    )
    ag_train["Train"] = (
        ag_train["Train_n"].astype(str) + " (" + ag_train["Train_%"].astype(str) + ")"
    )

    ag_test = pd.DataFrame(
        {
            "Test_n": dem[dem["dataset"] == "test"]["age_group"].value_counts(),
            "Test_%": round(
                (
                    dem[dem["dataset"] == "test"]["age_group"].value_counts()
                    / len(dem[dem["dataset"] == "test"]["dw_ek_borger"])
                )
                * 100,
                2,
            ),
        },
    )
    ag_test["Test"] = (
        ag_test["Test_n"].astype(str) + " (" + ag_test["Test_%"].astype(str) + ")"
    )

    ag = (
        pd.concat([ag_overall["Overall"], ag_train["Train"], ag_test["Test"]], axis=1)
        .reset_index()
        .rename(columns={"index": "Grouping"})
    )
    ag["Grouping"] = "Age group " + ag["Grouping"].astype(str) + ", n (%)"

    # Collect it all
    return pd.concat([n_prediction_days, n_adms, n_pts, n_sex_female, age_median, ag])


def get_sex_and_age_group_at_first_contact(df: pd.DataFrame) -> pd.DataFrame:
    # demographics
    dem = df[["dw_ek_borger", "pred_sex_female", "pred_age_in_years"]]
    dem = (
        dem.groupby(["dw_ek_borger", "pred_sex_female"])["pred_age_in_years"]
        .min()
        .reset_index()
    )

    age_groups = [18, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 120]

    labels = [
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
    ]

    dem["age_group"] = pd.cut(
        dem["pred_age_in_years"],
        bins=age_groups,
        labels=labels,
        right=False,
    )

    return dem
